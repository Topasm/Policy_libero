#!/usr/bin/env python3
"""
Training script for the Bidirectional Autore    # Create a normalizer that only handles the "observation.state" key
    # The bidirectional transformer model will internally map normalization
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, 
        cfg.normalization_mapping, 
        dataset_metadata.stats)ransformer.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import safetensors.torch
import numpy as np  # Added for stats saving
from tqdm import tqdm  # Added for progress bar
from datetime import datetime

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

from model.predictor.policy import (
    HierarchicalAutoregressivePolicy,
    compute_loss
)

from model.predictor.config import PolicyConfig
from model.predictor.bidirectional_dataset import BidirectionalTrajectoryDataset
from model.predictor.normalization_utils import KeyMappingNormalizer


def main():
    """Main training function."""
    # Instantiate unified config
    cfg = PolicyConfig()
    # Configuration
    output_directory = Path("outputs/train/bidirectional_transformer")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Update cfg with dataset dimensions below ---
    # --- Dataset Setup ---
    cfg.data.dataset_repo_id = "yongjincho/libero"   # <--- CHANGED
    dataset_metadata = LeRobotDatasetMetadata(cfg.data.dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    cfg.diffusion.input_features = {
        "observation.state": features["observation.state"]}
    # Dummy config to access properties
    cfg.diffusion.output_features = {
        "observation.image": features["observation.image"]}
    cfg.model.state_dim = features["observation.state"].shape[-1]
    cfg.model.input_features = {
        "observation.state": features["observation.state"]
    }
    cfg.model.output_features = {
        # Define what the model is expected to output, matching keys in compute_loss
        "predicted_forward_states": features["observation.state"],
        "predicted_goal_images": features["observation.image"],
        # Assuming this is an output
        "predicted_backward_states": features["observation.state"],
        # Placeholder, actual is latent dim
        "predicted_goal_latents": features["observation.image"],
    }

    state_dim = features["observation.state"].shape[-1]
    print(f"State dimension: {state_dim}")

    # We'll create normalizers later when setting up the model
    lerobot_dataset = LeRobotDataset(
        cfg.data.dataset_repo_id, delta_timestamps=None)  # Using base dataset

    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        forward_steps=32,
        backward_steps=32,
        image_key="observation.image",
        state_key="observation.state",
        n_obs_steps=3  # Enable temporal encoding
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,  # Important for consistent batch sizes
        collate_fn=BidirectionalTrajectoryDataset.collate_fn
    )

    # --- Create normalizers ---
    # Basic normalizer that works with "observation.state" key
    normalize_state_base = Normalize(
        {"observation.state": features["observation.state"]},
        cfg.normalization_mapping,
        dataset_metadata.stats)

    # Create a key mapping normalizer that maps from batch keys to normalizer keys
    key_mapping = {
        "initial_states": "observation.state",
        "forward_states": "observation.state",
        "backward_states": "observation.state"
    }

    # Wrap the base normalizer with our key mapper
    wrapped_normalizer = KeyMappingNormalizer(
        normalize_state_base, key_mapping)

    # Create the model without internal normalizers (we'll use external normalization)
    model = HierarchicalAutoregressivePolicy(config=cfg)
    model.to(device)
    model.train()  # Set model to training mode

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler can still be used if desired, adjusting T_max
    T_max = cfg.training.training_steps * cfg.training.lr_scheduler_T_max_mult
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max)

    # --- Training Loop ---
    print("Starting Training...")
    step = 0
    done = False

    while not done:
        for batch in tqdm(dataloader, desc=f"Training Step: {step}/{cfg.training.training_steps}"):
            # Normalize and move to device
            batch = wrapped_normalizer(batch)
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

            # <--- CHANGED: extract language instructions
            language_instructions = batch_device['language_instruction']

            optimizer.zero_grad()

            predictions = model(
                initial_images=batch_device['initial_images'],
                initial_states=batch_device['initial_states'],
                language_instruction=language_instructions  # <--- CHANGED
            )

            total_loss = compute_loss(predictions, batch_device)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step scheduler

            # Logging
            if step % cfg.training.log_freq == 0:
                print(
                    f"Step: {step}/{cfg.training.training_steps} | Loss: {total_loss.item():.4f}")

            # Checkpointing (similar to rtdiffusion)
            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= cfg.training.training_steps:
                done = True
                break

    # --- Save Final Model & Config ---
    final_path = output_directory / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to: {final_path}")

    # Save unified config
    cfg.save_pretrained(output_directory)

    # Save stats
    stats_to_save = {k: torch.from_numpy(
        v) for k, v in dataset_metadata.stats.items() if isinstance(v, np.ndarray)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")


if __name__ == "__main__":
    main()
