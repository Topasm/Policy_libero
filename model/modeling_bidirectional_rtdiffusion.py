import torch
import torch.nn as nn
from collections import deque

from lerobot.common.policies.normalize import Normalize, Unnormalize


class HierarchicalPolicy(nn.Module):
    """
    A policy wrapper for evaluation that implements per-step replanning.
    It manages observation history and calls the underlying model on every step.
    """

    def __init__(self, policy_model, dataset_stats, output_features, config):
        super().__init__()

        self.model = policy_model
        self.config = config
        # 이미 output_features를 저장하고 있습니다.
        self.output_features = output_features

        # Create normalizers here for inference
        d_cfg = self.config.data
        self.normalize_inputs = Normalize(
            d_cfg.input_features, d_cfg.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(
            self.output_features, d_cfg.normalization_mapping, dataset_stats)

        # These are not model parameters, so they won't be moved by .to(device), which is correct.
        self.observation_queue = deque(maxlen=d_cfg.n_obs_steps)
        self.action_queue = deque()
        self.reset()

    def reset(self):
        """Resets the observation history queue for a new episode."""
        self.observation_queue.clear()
        self.action_queue.clear()

    @torch.no_grad()
    def select_action(self, observation_dict: dict) -> torch.Tensor:
        """
        Selects an action by performing a full planning pass on every call.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 1. Preprocess observation and add to history queue
        # Ensure input tensors are on the same device as the normalizers
        for key, value in observation_dict.items():
            if isinstance(value, torch.Tensor):
                observation_dict[key] = value.unsqueeze(0).to(device)

        normalized_obs = self.normalize_inputs(observation_dict)

        self.observation_queue.append({
            "observation.image": normalized_obs["observation.image"].squeeze(0),
            "observation.state": normalized_obs["observation.state"].squeeze(0),
        })

        # 2. Return a dummy action if history is not yet full
        if len(self.observation_queue) < self.config.data.n_obs_steps:
            # [FIXED] unnormalize_outputs 대신, 클래스에 저장된 self.output_features를 사용합니다.
            action_dim = self.output_features["action"].shape[-1]
            return torch.zeros(action_dim, device=device)

        # 3. Prepare model input from the full observation history
        model_input_batch = {
            "initial_images": torch.stack([obs["observation.image"]
                                           for obs in self.observation_queue]).unsqueeze(0),
            "initial_states": torch.stack([obs["observation.state"]
                                           for obs in self.observation_queue]).unsqueeze(0),
            "language_instruction": observation_dict["language_instruction"]
        }

        # 4. Perform a full forward pass to get a new action plan
        predictions = self.model.forward(**model_input_batch)
        actions_normalized = predictions['predicted_forward_actions']

        # 5. Take only the FIRST action from the predicted plan
        # Take the first action chunk (length 1)
        next_action_normalized = actions_normalized[:, :1, :]

        # 6. Denormalize the single action
        action_denormalized = self.unnormalize_outputs(
            {"action": next_action_normalized})["action"]

        # Return a single action tensor
        return action_denormalized.squeeze(0).squeeze(0)
