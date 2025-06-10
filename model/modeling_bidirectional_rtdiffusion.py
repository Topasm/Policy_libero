import torch
from torch import nn
from collections import deque
import numpy as np
from einops import rearrange

from lerobot.common.policies.normalize import Normalize, Unnormalize


class HierarchicalPolicy(nn.Module):
    def __init__(
        self,
        bidirectional_transformer: nn.Module,
        inverse_dynamics_model: nn.Module,
        dataset_stats,
        all_dataset_features,
        n_obs_steps,
        output_features,
    ):
        super().__init__()
        self.bidirectional_transformer = bidirectional_transformer
        self.config = bidirectional_transformer.config
        self.inverse_dynamics_model = inverse_dynamics_model
        self.n_obs_steps = n_obs_steps
        self.output_features = output_features

        self.normalize_inputs = Normalize(
            self.config.data.input_features,
            self.config.data.normalization_mapping,
            dataset_stats
        )

        self.unnormalize_outputs = Unnormalize(
            output_features,
            self.config.data.normalization_mapping,
            dataset_stats
        )

        self.reset()

    def reset(self):
        self.observation_queue = deque(maxlen=self.n_obs_steps)

    @torch.no_grad()
    def select_action(self, current_raw_observation: dict) -> torch.Tensor:
        # Expected input keys: 'observation.image', 'observation.state', 'language_instruction'
        for key in current_raw_observation:
            if isinstance(current_raw_observation[key], torch.Tensor) and current_raw_observation[key].ndim in [1, 3]:
                current_raw_observation[key] = current_raw_observation[key].unsqueeze(
                    0)

        normalized_obs = self.normalize_inputs(current_raw_observation)

        obs_for_queue = {
            "observation.image": normalized_obs["observation.image"][0],
            "observation.state": normalized_obs["observation.state"][0],
        }
        self.observation_queue.append(obs_for_queue)

        if len(self.observation_queue) < self.n_obs_steps:
            # Not enough history yet, return a default/zero action
            # --- [FIXED] Use the stored self.output_features ---
            action_dim = self.output_features["action"].shape[-1]
            return torch.zeros(action_dim, device=next(self.parameters()).device)

        # Prepare batch for the policy model
        model_input_batch = {
            "initial_images": torch.stack([obs["observation.image"] for obs in self.observation_queue]).unsqueeze(0),
            "initial_states": torch.stack([obs["observation.state"] for obs in self.observation_queue]).unsqueeze(0),
            "language_instruction": current_raw_observation["language_instruction"]
        }

        state_plan = self._generate_state_plan(model_input_batch)
        actions = self._generate_actions_from_states(state_plan)

        # [FIXED] Call the unnormalize object like a function
        denormalized_action = self.unnormalize_outputs(
            {"action": actions[:, 0]})["action"]

        return denormalized_action.squeeze(0)

    def _generate_state_plan(self, model_input_batch: dict) -> torch.Tensor:
        """Generates a sequence of future states using the planner."""
        predictions = self.bidirectional_transformer.generate(
            **model_input_batch)
        return predictions['predicted_forward_states']

    def _generate_actions_from_states(self, state_plan: torch.Tensor) -> torch.Tensor:
        """Generates a sequence of actions from a state plan using the inverse dynamics model."""
        actions = []
        for i in range(state_plan.size(1) - 1):
            s_t = state_plan[:, i, :]
            s_t_plus_1 = state_plan[:, i + 1, :]

            # Call the model with two separate tensors, just like in training
            action = self.inverse_dynamics_model(s_t, s_t_plus_1)

            actions.append(action)
        return torch.stack(actions, dim=1)
