import torch
from torch import nn
from collections import deque
import numpy as np

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

        self.observation_queue = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()

        self.reset()

    def reset(self):
        """Reset observation and action queues."""
        self.observation_queue.clear()
        self.action_queue.clear()

    @torch.no_grad()
    def select_action(self, current_raw_observation: dict) -> torch.Tensor:
        """
        Selects an action using an action chunking strategy.
        A new plan is generated only when the action queue is empty.
        """
        # Add a batch dimension if not present
        for key in current_raw_observation:
            if isinstance(current_raw_observation[key], torch.Tensor) and current_raw_observation[key].ndim in [1, 3]:
                current_raw_observation[key] = current_raw_observation[key].unsqueeze(
                    0)

        # Normalize the new observation and add it to the history queue
        normalized_obs = self.normalize_inputs(current_raw_observation)

        obs_for_queue = {
            "observation.image": normalized_obs["observation.image"][0],
            "observation.state": normalized_obs["observation.state"][0],
        }
        self.observation_queue.append(obs_for_queue)

        # Return a default zero action if there isn't enough history yet
        if len(self.observation_queue) < self.n_obs_steps:
            action_dim = self.output_features["action"].shape[-1]
            return torch.zeros(action_dim, device=next(self.parameters()).device)

        # Generate a new action plan only if the action queue is empty
        if not self.action_queue:
            # Prepare a batch from the history queue
            model_input_batch = {
                "initial_images": torch.stack([obs["observation.image"] for obs in self.observation_queue]).unsqueeze(0),
                "initial_states": torch.stack([obs["observation.state"] for obs in self.observation_queue]).unsqueeze(0),
                "language_instruction": current_raw_observation["language_instruction"]
            }

            # 1. Generate a plan of future states
            state_plan = self._generate_state_plan(model_input_batch)

            # 2. Generate a sequence of actions from the state plan
            actions_normalized = self._generate_actions_from_states(state_plan)

            # 3. Denormalize the entire action sequence
            actions_denormalized = self.unnormalize_outputs(
                {"action": actions_normalized})["action"]

            # 4. Add the sequence of actions to the execution queue
            # Squeeze the batch dimension and add each action tensor to the deque
            for i in range(actions_denormalized.shape[1]):
                self.action_queue.append(actions_denormalized.squeeze(0)[i])

        # Pop the next action from the queue and return it
        return self.action_queue.popleft()

    def _generate_state_plan(self, model_input_batch: dict) -> torch.Tensor:
        """Generates a sequence of future states using the planner."""
        predictions = self.bidirectional_transformer.generate(
            **model_input_batch)
        return predictions['predicted_forward_states']

    def _generate_actions_from_states(self, state_plan: torch.Tensor) -> torch.Tensor:
        """
        Generates a shorter sequence of actions (action chunk) from the state plan.
        """
        # --- [FIXED] Use only a short horizon of the plan for action generation ---
        # The planner generates a long state plan (e.g., 32 steps), but we only
        # convert a shorter chunk into actions to be queued. This is more stable.
        # Get horizon from config (e.g., 8)
        action_horizon = self.config.data.n_action_steps

        # We need action_horizon+1 states to generate action_horizon actions (a_0 to a_{H-1})
        # This requires states s_0 to s_H.
        state_plan_chunk = state_plan[:, :action_horizon + 1]
        # --- END FIX ---

        actions = []
        # Loop over the shorter state plan chunk
        # This will create pairs (s_0, s_1), (s_1, s_2), ..., (s_{H-1}, s_H)
        for i in range(state_plan_chunk.size(1) - 1):
            s_t = state_plan_chunk[:, i, :]
            s_t_plus_1 = state_plan_chunk[:, i + 1, :]
            action = self.inverse_dynamics_model(s_t, s_t_plus_1)
            actions.append(action)

        return torch.stack(actions, dim=1)
