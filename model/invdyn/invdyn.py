import torch
from torch import nn, Tensor


class MlpInvDynamic(nn.Module):
    """
    An inverse dynamics model inspired by the Seer action decoder,
    using a shared body and separate heads for arm and gripper actions.
    """

    def __init__(self, o_dim, a_dim, hidden_dim=512, dropout=0.1, use_layernorm=True, out_activation: nn.Module = nn.Tanh()):
        super().__init__()

        if a_dim != 7:
            raise ValueError(
                f"This model architecture expects a 7D action space (6 arm + 1 gripper), but got {a_dim}")

        # Input dimension is doubled because it takes a pair of states (s_t, s_{t+1})
        input_dim = o_dim * 2

        # 1. Shared Body: learns common features from the state transition.
        # This part is shared by both arm and gripper predictions.
        body_hidden_dim = hidden_dim // 2
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, body_hidden_dim),
            nn.LayerNorm(body_hidden_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
        )

        # 2. Separate Heads: predict arm and gripper actions from the shared features.
        # Arm action head
        self.arm_head = nn.Sequential(
            nn.Linear(body_hidden_dim, 6),
            out_activation  # Tanh to output values between -1 and 1
        )

        # Gripper action head (outputs logits, no activation)
        self.gripper_head = nn.Linear(body_hidden_dim, 1)

    def forward(self, s_t: Tensor, s_t_plus_1: Tensor) -> Tensor:
        # Concatenate state pairs to create the input
        state_pair = torch.cat([s_t, s_t_plus_1], dim=-1)

        # The input shape is likely (B, T, D_pair), flatten it for the MLP
        original_shape = state_pair.shape
        if state_pair.ndim == 3:
            state_pair = state_pair.reshape(-1, original_shape[-1])

        # --- Main Logic ---
        # 1. Process the input through the shared body to get common features
        shared_features = self.shared_body(state_pair)

        # 2. Use the shared features to predict with separate heads
        pred_arm_action = self.arm_head(shared_features)
        pred_gripper_logit = self.gripper_head(shared_features)

        # Apply sign() to gripper logit only during evaluation for binary output
        if not self.training:
            pred_gripper_action = torch.sign(pred_gripper_logit)
        else:
            # During training, we need the raw logit for BCEWithLogitsLoss
            pred_gripper_action = pred_gripper_logit

        # 3. Concatenate outputs to form the final 7D action
        final_action = torch.cat(
            [pred_arm_action, pred_gripper_action], dim=-1)

        # Reshape back to original sequence shape if necessary
        if len(original_shape) == 3:
            final_action = final_action.view(
                original_shape[0], original_shape[1], -1)

        return final_action
