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
import wandb  # [MODIFIED] Import wandb

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features, PolicyFeature
from lerobot.common.policies.normalize import Normalize
from lerobot.configs.types import FeatureType

from model.predictor.policy import HierarchicalAutoregressivePolicy, compute_loss
from config.config import PolicyConfig
from dataset.bidirectional_dataset import BidirectionalTrajectoryDataset
from model.predictor.normalization_utils import KeyMappingNormalizer


def main():
    """Main training function."""
    cfg = PolicyConfig()
    output_directory = Path("outputs/train/bidirectional_transformere")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- [MODIFIED] Wandb Initialization ---
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=cfg.to_dict()  # Log all hyperparameters
    )
    # ---

    # --- Dataset and Config Setup ---
    dataset_metadata = LeRobotDatasetMetadata(cfg.data.dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    cfg.hierarchical_transformer.state_dim = features["observation.state"].shape[-1]

    cfg.data.input_features = {
        "observation.state": features["observation.state"]
    }

    # --- FIXED: Correct way to define output features ---
    # Get shape from an example image key
    example_key = cfg.data.image_keys[0]
    # The shape from metadata is typically (C, H, W)
    c, h, w = features[example_key].shape
    combined_c = c * len(cfg.data.image_keys)

    # [MODIFIED] Update output_features to reflect action prediction
    cfg.data.output_features = {
        # Changed from states to actions
        "predicted_forward_actions": features["action"],
        "predicted_backward_states": features["observation.state"],
        # Single 3-channel image
        "predicted_goal_images": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    # --- END FIX ---

    print(f"State dimension set to: {cfg.hierarchical_transformer.state_dim}")
    print(f"Combined image channels set to: {combined_c}")

    lerobot_dataset = LeRobotDataset(
        cfg.data.dataset_repo_id, delta_timestamps=None)

    # [MODIFIED] Get the tasks mapping from the dataset metadata.
    tasks_mapping = dataset_metadata.tasks
    print(f"Loaded {len(tasks_mapping)} tasks from dataset metadata.")

    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        forward_steps=cfg.hierarchical_transformer.forward_steps,
        backward_steps=cfg.hierarchical_transformer.backward_steps,
        n_obs_steps=cfg.data.n_obs_steps,
        image_keys=cfg.data.image_keys,
        state_key="observation.state",
        # [MODIFIED] Pass the tasks mapping to the dataset class.
        tasks=tasks_mapping,
        is_train=True
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
    normalize_state_base = Normalize(
        {"observation.state": features["observation.state"]},
        cfg.data.normalization_mapping,
        dataset_metadata.stats
    )
    # [MODIFIED] Remove 'forward_states' from the normalization key mapping
    key_mapping = {
        "initial_states": "observation.state",
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

            batch = wrapped_normalizer(batch)
            batch_device = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            language_instructions = batch.get('language_instruction', [])

            optimizer.zero_grad()

            predictions = model(
                initial_images=batch_device['initial_images'],
                initial_states=batch_device['initial_states'],
                language_instruction=language_instructions
            )

            # Get all losses including individual components
            losses = compute_loss(predictions, batch_device)
            # Extract the total loss for backward pass
            total_loss = losses['total_loss']
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # --- [MODIFIED] Wandb Logging ---
            if step % cfg.training.log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                loss_value = total_loss.item()

                # Prepare log dictionary with all individual losses
                log_dict = {
                    "train/total_loss": loss_value,
                    "train/learning_rate": current_lr,
                    "step": step
                }

                # Add individual loss components to the log dictionary
                for loss_name, loss_tensor in losses.items():
                    if loss_name != 'total_loss':  # Skip total_loss as we've already logged it
                        log_dict[f"train/{loss_name}"] = loss_tensor.item()

                # Log all metrics to wandb
                wandb.log(log_dict)

                progress_bar.set_postfix(
                    {"Loss": f"{loss_value:.4f}", "LR": f"{current_lr:.6f}"})

            # Log prediction/ground truth images periodically
            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                tqdm.write(f"\nSaved checkpoint: {ckpt_path}")

                # --- [MODIFIED] Simplified Image Logging Logic ---
                # The model predicts a single 3-channel image.
                pred_img_front = predictions['predicted_goal_images'][0].cpu().clamp(
                    0, 1)

                # Get the ground truth images for reference.
                true_img_front = batch_device['goal_images'][0, 0].cpu()
                true_img_wrist = batch_device['goal_images'][0, 1].cpu()

                # Log only the predicted front view.
                wandb.log({
                    "predictions/goal_image_front": wandb.Image(pred_img_front),
                    "ground_truth/goal_image_front": wandb.Image(true_img_front),
                    # For reference
                    "ground_truth/goal_image_wrist": wandb.Image(true_img_wrist),
                    "step": step
                })
                # --- END FIX ---

            step += 1
            progress_bar.update(1)

    progress_bar.close()

    # --- Save Final Model & Config ---
    final_path = output_directory / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to: {final_path}")

    cfg.save_pretrained(output_directory)

    # --- FIXED: Correctly flatten the stats dictionary for safetensors ---
    stats_to_save = {}
    for key, value_dict in dataset_metadata.stats.items():
        if isinstance(value_dict, dict):
            for stat_key, stat_value in value_dict.items():
                if isinstance(stat_value, torch.Tensor):
                    # Create a flattened key, e.g., "action.min"
                    flattened_key = f"{key}.{stat_key}"
                    stats_to_save[flattened_key] = stat_value
    # --- END FIX ---

    if stats_to_save:
        safetensors.torch.save_file(
            stats_to_save, output_directory / "stats.safetensors")
        print(f"Stats saved to: {output_directory / 'stats.safetensors'}")

    wandb.finish()  # [MODIFIED] Finish the wandb run


if __name__ == "__main__":
    main()
