#!/usr/bin/env python3
"""
Dataset wrapper for the Bidirectional Autoregressive Transformer.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import torchvision.transforms as T

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class BidirectionalTrajectoryDataset(Dataset):
    """
    A dataset wrapper for bidirectional model training. It prepares data by:
    - Loading a history of observations (images and states).
    - Loading a sequence of future actions for end-to-end prediction.
    - Loading a sequence of backward states from the goal.
    - Loading a final goal image.
    - Applying data augmentation during training.
    - Handling padding for variable-length episodes.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        tasks: Dict[str, str],
        forward_steps: int,
        backward_steps: int,
        n_obs_steps: int,
        image_keys: List[str],
        state_key: str,
        is_train: bool = False
    ):
        super().__init__()
        self.lerobot_dataset = lerobot_dataset
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.n_obs_steps = n_obs_steps
        self.image_keys = image_keys
        self.state_key = state_key
        self.tasks = tasks
        self.is_train = is_train

        # Define a resize transform that will be applied to all images for consistent sizing.
        self.resize = T.Resize((224, 224), antialias=True)

        if self.is_train:
            # Define augmentation pipelines that will be applied AFTER resizing.
            self.main_camera_aug = T.Compose([
                T.RandomCrop(size=(224, 224), padding=int(
                    224 * 0.05), padding_mode='edge'),
                T.RandomRotation(degrees=5),
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ])
            self.wrist_camera_aug = T.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.5)

        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} bidirectional trajectory samples.")

    def _create_samples(self):
        """Pre-computes all possible valid start indices for trajectory sampling."""
        samples = []
        episode_data_index = self.lerobot_dataset.episode_data_index
        if not (episode_data_index and 'from' in episode_data_index and 'to' in episode_data_index):
            raise ValueError(
                "`episode_data_index` is missing or malformed in the LeRobotDataset.")

        for episode_idx in range(len(episode_data_index['from'])):
            from_idx = episode_data_index['from'][episode_idx].item()
            to_idx = episode_data_index['to'][episode_idx].item() - 1
            episode_length = to_idx - from_idx + 1

            min_start_offset = self.n_obs_steps - 1
            if episode_length <= min_start_offset:
                continue

            for start_offset in range(min_start_offset, episode_length):
                samples.append({
                    'episode_idx': episode_idx,
                    'start_idx': from_idx + start_offset,
                    'episode_true_end_idx': to_idx,
                    'episode_abs_start_idx': from_idx,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        start_idx = sample_info['start_idx']
        end_idx = sample_info['episode_true_end_idx']
        abs_start_idx = sample_info['episode_abs_start_idx']

        # 1. Load history observations
        initial_images, initial_states = [], []
        for i in range(self.n_obs_steps):
            obs_idx = start_idx - (self.n_obs_steps - 1 - i)
            obs_idx = max(obs_idx, abs_start_idx)
            obs_data = self.lerobot_dataset[obs_idx]

            front_img = torch.as_tensor(
                obs_data[self.image_keys[0]], dtype=torch.float32)
            wrist_img = torch.as_tensor(
                obs_data[self.image_keys[1]], dtype=torch.float32)

            front_img, wrist_img = self.resize(
                front_img), self.resize(wrist_img)
            if self.is_train:
                front_img = self.main_camera_aug(front_img)
                wrist_img = self.wrist_camera_aug(wrist_img)

            initial_images.append(torch.stack([front_img, wrist_img]))
            initial_states.append(torch.as_tensor(
                obs_data[self.state_key], dtype=torch.float32))

        # 2. Load goal images (only resized, not augmented)
        goal_data = self.lerobot_dataset[end_idx]
        goal_front = self.resize(torch.as_tensor(
            goal_data[self.image_keys[0]], dtype=torch.float32))
        goal_wrist = self.resize(torch.as_tensor(
            goal_data[self.image_keys[1]], dtype=torch.float32))
        goal_images = torch.stack([goal_front, goal_wrist])

        # 3. Load future trajectories and handle padding
        forward_actions, action_paddings, backward_states = [], [], []

        # Load forward actions
        for j in range(self.forward_steps):
            fwd_idx = start_idx + j
            is_pad = fwd_idx > end_idx
            load_idx = min(fwd_idx, end_idx)
            fwd_data = self.lerobot_dataset[load_idx]

            forward_actions.append(torch.as_tensor(
                fwd_data['action'], dtype=torch.float32))
            action_paddings.append(torch.tensor(is_pad, dtype=torch.bool))

        # Load backward states
        for i in range(self.backward_steps):
            bwd_idx = end_idx - i
            load_idx = max(bwd_idx, abs_start_idx)
            bwd_data = self.lerobot_dataset[load_idx]
            backward_states.append(torch.as_tensor(
                bwd_data[self.state_key], dtype=torch.float32))

        backward_states.reverse()

        # 4. Load metadata
        task_id_data = self.lerobot_dataset[start_idx]
        task_id = task_id_data.get('task_id', torch.tensor(0)).item()
        task_description = self.tasks.get(str(task_id), "")

        episode_length = end_idx - abs_start_idx
        current_step = start_idx - abs_start_idx
        normalized_timestep = float(
            current_step) / float(episode_length + 1e-6)

        return {
            'initial_images': torch.stack(initial_images),
            'initial_states': torch.stack(initial_states),
            'goal_images': goal_images,
            # Key is now forward_actions
            'forward_actions': torch.stack(forward_actions),
            'backward_states': torch.stack(backward_states),
            'action_is_pad': torch.stack(action_paddings),
            'normalized_timestep': torch.tensor(normalized_timestep, dtype=torch.float32),
            'language_instruction': task_description,
        }

    @staticmethod
    def collate_fn(batch):
        result = {}
        if not batch:
            return result
        first_item_keys = batch[0].keys()
        for key in first_item_keys:
            key_items = [item[key] for item in batch if key in item]
            if not key_items:
                continue
            if isinstance(key_items[0], torch.Tensor):
                result[key] = torch.stack(key_items)
            else:
                result[key] = key_items
        return result
