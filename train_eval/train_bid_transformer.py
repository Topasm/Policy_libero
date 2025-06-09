#!/usr/bin/env python3
"""
Training script for the Bidirectional Autoregressive Transformer.
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import safetensors.torch
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

from model.predictor.policy import HierarchicalAutoregressivePolicy, compute_loss
from model.predictor.config import PolicyConfig
from model.predictor.bidirectional_dataset import BidirectionalTrajectoryDataset
from model.predictor.normalization_utils import KeyMappingNormalizer


def main():
    """Main training function."""
    # Instantiate unified config
    cfg = PolicyConfig()
    output_directory = Path("outputs/train/bidirectional_transformer")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset and Config Setup ---
    # Load metadata and features first to update the config
    dataset_metadata = LeRobotDatasetMetadata(cfg.data.dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Update config with actual dimensions from the dataset
    cfg.hierarchical_transformer.state_dim = features["observation.state"].shape[-1]

    # Set feature definitions in the data config
    cfg.data.input_features = {
        "observation.state": features["observation.state"]
    }
    cfg.data.output_features = {
        "predicted_forward_states": features["observation.state"],
        "predicted_backward_states": features["observation.state"],
        "predicted_goal_images": features["observation.image"],
    }

    print(f"State dimension set to: {cfg.hierarchical_transformer.state_dim}")

    # Use the base dataset to create our custom bidirectional dataset
    lerobot_dataset = LeRobotDataset(
        cfg.data.dataset_repo_id, delta_timestamps=None)

    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        forward_steps=cfg.hierarchical_transformer.forward_steps,
        backward_steps=cfg.hierarchical_transformer.backward_steps,
        n_obs_steps=cfg.data.n_obs_steps,
        image_key="observation.image",
        state_key="observation.state"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=BidirectionalTrajectoryDataset.collate_fn
    )

    # --- Normalization Setup ---
    # Create a base normalizer that understands "observation.state"
    normalize_state_base = Normalize(
        {"observation.state": features["observation.state"]},
        cfg.data.normalization_mapping,
        dataset_metadata.stats
    )

    # Map the keys from our batch to the keys the normalizer expects
    key_mapping = {
        "initial_states": "observation.state",
        "forward_states": "observation.state",
        "backward_states": "observation.state"
    }
    wrapped_normalizer = KeyMappingNormalizer(
        normalize_state_base, key_mapping)

    # --- Model, Optimizer, Scheduler ---
    model = HierarchicalAutoregressivePolicy(config=cfg)
    model.to(device)
    model.train()

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

    T_max = cfg.training.training_steps * cfg.training.lr_scheduler_T_max_mult
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max)

    # --- Training Loop ---
    print("Starting Training...")
    step = 0
    done = False
    progress_bar = tqdm(total=cfg.training.training_steps, desc="Training")

    while not done:
        for batch in dataloader:
            if step >= cfg.training.training_steps:
                done = True
                break

            # Normalize data first, then move to device
            batch = wrapped_normalizer(batch)
            batch_device = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # The language instruction might be a list of strings, kept on CPU
            language_instructions = batch.get('language_instruction', [])

            optimizer.zero_grad()

            predictions = model(
                initial_images=batch_device['initial_images'],
                initial_states=batch_device['initial_states'],
                language_instruction=language_instructions
            )

            total_loss = compute_loss(predictions, batch_device)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % cfg.training.log_freq == 0:
                print(
                    f"\nStep: {step}/{cfg.training.training_steps} | Loss: {total_loss.item():.4f}")

            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

            step += 1
            progress_bar.update(1)

    progress_bar.close()

    # --- Save Final Model & Config ---
    final_path = output_directory / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to: {final_path}")

    cfg.save_pretrained(output_directory)

    stats_to_save = {k: torch.from_numpy(
        v) for k, v in dataset_metadata.stats.items() if isinstance(v, np.ndarray)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Stats saved to: {output_directory / 'stats.safetensors'}")


if __name__ == "__main__":
    main()
