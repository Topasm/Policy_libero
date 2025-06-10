import torch
from torch import nn, Tensor


class _SingleMlpInvDyn(nn.Module):
    """A simple MLP for inverse dynamics, used as a building block."""

    def __init__(self, o_dim, a_dim, hidden_dim=512, dropout=0.1, use_layernorm=True, out_activation: nn.Module = nn.Tanh()):
        super().__init__()
        # 입력은 s_t와 s_{t+1}이 합쳐진 형태이므로 입력 차원은 o_dim * 2 입니다.
        input_dim = o_dim * 2

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, a_dim),
            out_activation
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # [FIXED] Handle inputs with a sequence dimension (e.g., [B, T, D_in])
        original_shape = x.shape
        if x.ndim == 3:
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, original_shape[-1])

        out_flat = self.net(x)

        # Reshape back to original sequence shape if necessary
        if len(original_shape) == 3:
            out = out_flat.view(original_shape[0], original_shape[1], -1)
        else:
            out = out_flat

        return out


class SeparatedInvDyn(nn.Module):
    """
    Inverse dynamics model that uses two separate MLPs with different inputs.
    """

    def __init__(self, o_dim, a_dim, hidden_dim=512, dropout=0.1, use_layernorm=True, out_activation: nn.Module = nn.Tanh()):
        super().__init__()
        if a_dim != 7:
            raise ValueError(
                f"SeparatedInvDyn expects total action dim of 7, but got {a_dim}")

        arm_o_dim = o_dim
        gripper_o_dim = 1

        self.arm_model = _SingleMlpInvDyn(
            o_dim=arm_o_dim, a_dim=6, hidden_dim=hidden_dim, dropout=dropout,
            use_layernorm=use_layernorm, out_activation=out_activation
        )
        self.gripper_model = _SingleMlpInvDyn(
            o_dim=gripper_o_dim, a_dim=1, hidden_dim=hidden_dim // 4,
            dropout=dropout, use_layernorm=use_layernorm, out_activation=out_activation
        )

    def forward(self, s_t: Tensor, s_t_plus_1: Tensor) -> Tensor:
        state_pair_arm = torch.cat([s_t, s_t_plus_1], dim=-1)
        pred_arm_action = self.arm_model(state_pair_arm)

        s_t_gripper = s_t[..., -1:]
        s_t_plus_1_gripper = s_t_plus_1[..., -1:]
        state_pair_gripper = torch.cat(
            [s_t_gripper, s_t_plus_1_gripper], dim=-1)
        pred_gripper_action = self.gripper_model(state_pair_gripper)

        # --- [MODIFIED] Apply sign() function only during evaluation ---
        if not self.training:
            pred_gripper_action = torch.sign(pred_gripper_action)
        # --- END MODIFICATION ---

        return torch.cat([pred_arm_action, pred_gripper_action], dim=-1)


# 이전 MlpInvDynamic과의 호환성을 위해 별칭을 유지합니다.
MlpInvDynamic = SeparatedInvDyn
