import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
import safetensors.torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.invdyn.invdyn import MlpInvDynamic
from model.predictor.config import PolicyConfig


def compute_loss(batch: dict[str, Tensor], model: MlpInvDynamic, cfg: PolicyConfig) -> Tensor:
    """
    Computes the inverse dynamics loss for the separated model.
    """
    all_states = batch["observation.state"]
    s_t = all_states[:, :-1]
    s_t_plus_1 = all_states[:, 1:]
    true_actions = batch["action"]

    # The new model expects separate s_t and s_{t+1}
    pred_actions = model(s_t, s_t_plus_1)

    num_actions = min(pred_actions.shape[1], true_actions.shape[1])
    pred_actions = pred_actions[:, :num_actions]
    true_actions = true_actions[:, :num_actions]

    if "action_is_pad" in batch:
        pad_mask = batch["action_is_pad"][:, :num_actions]
        mask = ~pad_mask

        loss_per_element = F.mse_loss(
            pred_actions, true_actions, reduction="none")
        loss_per_element = loss_per_element * mask.unsqueeze(-1)
        loss = loss_per_element.sum() / (mask.sum() *
                                         pred_actions.shape[-1] + 1e-8)
    else:
        loss = F.mse_loss(pred_actions, true_actions)

    return loss


def main():
    output_directory = Path("outputs/train/invdyn_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PolicyConfig()
    dataset_repo_id = cfg.data.dataset_repo_id
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    if cfg.inverse_dynamics.out_activation == "Tanh":
        activation_module = nn.Tanh()
    else:
        activation_module = nn.Identity()

    # [MODIFIED] Model Instantiation
    # MlpInvDynamic is now an alias for SeparatedInvDyn, which handles the split internally.
    invdyn_model = MlpInvDynamic(
        o_dim=features["observation.state"].shape[0],
        a_dim=features["action"].shape[0],  # a_dim is still 7 here
        hidden_dim=cfg.inverse_dynamics.hidden_dim,
        dropout=cfg.inverse_dynamics.dropout,
        use_layernorm=cfg.inverse_dynamics.use_layernorm,
        out_activation=activation_module,
    )
    invdyn_model.train()
    invdyn_model.to(device)

    normalize_state = Normalize(
        {"observation.state": features["observation.state"]
         }, cfg.data.normalization_mapping, dataset_metadata.stats
    )
    normalize_action = Normalize(
        {"action": features["action"]
         }, cfg.data.normalization_mapping, dataset_metadata.stats
    )

    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.data.state_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.data.action_delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

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

    step = 0
    done = False
    print("Starting Inverse Dynamics Model Training...")

    while not done:
        for batch in dataloader:
            if step >= cfg.training.training_steps:
                done = True
                break

            normalized_batch = normalize_state(batch)
            normalized_batch = normalize_action(normalized_batch)
            normalized_batch['action_is_pad'] = batch['action_is_pad']
            device_batch = {k: v.to(device) for k, v in normalized_batch.items(
            ) if isinstance(v, torch.Tensor)}

            loss = compute_loss(device_batch, invdyn_model, cfg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                invdyn_model.parameters(), max_norm=1.0, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % cfg.training.log_freq == 0:
                print(
                    f"Step: {step}/{cfg.training.training_steps} Loss: {loss.item():.4f}", flush=True)

            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"invdyn_step_{step}.pth"
                torch.save(invdyn_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

            step += 1

    final_path = output_directory / "invdyn_final.pth"
    torch.save(invdyn_model.state_dict(), final_path)
    print(
        f"\nTraining finished. Final model saved to: {final_path}", flush=True)

    cfg.save_pretrained(output_directory)

    stats_to_save = {}
    for key, value_dict in dataset_metadata.stats.items():
        if isinstance(value_dict, dict):
            for stat_key, stat_value in value_dict.items():
                if isinstance(stat_value, torch.Tensor):
                    stats_to_save[f"{key}.{stat_key}"] = stat_value

    if stats_to_save:
        safetensors.torch.save_file(
            stats_to_save, output_directory / "stats.safetensors")
        print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
