import torch
import torch.nn as nn  # Added to support activation function mapping
from pathlib import Path
import safetensors.torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.invdyn.invdyn import MlpInvDynamic
from model.predictor.config import PolicyConfig
#


def main():
    output_directory = Path("outputs/train/invdyn_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Unified Config Setup ---
    cfg = PolicyConfig()

    # --- Dataset and Metadata ---
    dataset_repo_id = cfg.data.dataset_repo_id
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # --- Activation Function Mapping ---
    # Convert string from config to an actual nn.Module
    if cfg.inverse_dynamics.out_activation == "Tanh":
        activation_module = nn.Tanh()
    else:
        activation_module = nn.Identity()

    # --- Model ---
    invdyn_model = MlpInvDynamic(
        o_dim=features["observation.state"].shape[0],
        a_dim=features["action"].shape[0],
        hidden_dim=cfg.inverse_dynamics.hidden_dim,
        dropout=cfg.inverse_dynamics.dropout,
        use_layernorm=cfg.inverse_dynamics.use_layernorm,
        out_activation=activation_module,  # Pass the nn.Module instance
    )
    invdyn_model.train()
    invdyn_model.to(device)

    # Helper model instance to reuse loss computation logic (as in original code)
    loss_computer = MyDiffusionModel(cfg).to(device)

    # --- Normalization ---
    # Access normalization_mapping from the correct sub-config 'data'
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]
         }, cfg.data.normalization_mapping, dataset_metadata.stats
    )
    normalize_action = Normalize(
        {"action": features["action"]
         }, cfg.data.normalization_mapping, dataset_metadata.stats
    )

    # --- Dataset ---
    # Fetch data using delta indices from the config file
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.data.state_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.data.action_delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer, Scheduler & Dataloader ---
    # Use training parameters from the config
    optimizer = torch.optim.Adam(
        invdyn_model.parameters(), lr=cfg.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.training_steps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Inverse Dynamics Model Training...")
    print(
        f"Training for {cfg.training.training_steps} steps with cosine annealing LR schedule")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    while not done:
        for batch in dataloader:
            if step >= cfg.training.training_steps:
                done = True
                break

            invdyn_loss_batch = normalize_state(batch)
            invdyn_loss_batch = normalize_action(invdyn_loss_batch)
            invdyn_loss_batch['action_is_pad'] = batch['action_is_pad']
            invdyn_loss_batch = {k: v.to(
                device) for k, v in invdyn_loss_batch.items() if isinstance(v, torch.Tensor)}

            loss = loss_computer.compute_invdyn_loss(
                invdyn_loss_batch, invdyn_model)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % cfg.training.log_freq == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"Step: {step}/{cfg.training.training_steps} Loss: {loss.item():.4f} LR: {current_lr:.6f}")

            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"invdyn_step_{step}.pth"
                torch.save(invdyn_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1

    # --- Save Final Model & Stats ---
    final_path = output_directory / "invdyn_final.pth"
    torch.save(invdyn_model.state_dict(), final_path)
    print(
        f"\nTraining finished. Final inverse dynamics model saved to: {final_path}")

    cfg.save_pretrained(output_directory)

    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items() if isinstance(v, torch.Tensor)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
