#!/usr/bin/env python3
"""
Bidirectional Autoregressive Transformer for image-conditioned trajectory generation.

This model implements the following pipeline with SOFT GOAL CONDITIONING and GLOBAL HISTORY CONDITIONING:
1. Input: sequence of initial images i_{t-k:t} and states st_{t-k:t} (n_obs_steps history)
2. Encode and flatten history into a single global_history_condition_embedding.
3. Using this global_history_condition_embedding:
    a. Generate goal image i_n (first prediction)
    b. Generate backward states st_n ... (conditioned on global history + goal)
    c. Generate forward states st_0 ... (conditioned on global history + goal + backward path)

The new prediction order (goal → backward → forward) enables soft conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from model.modules.custom_transformer import RMSNorm, ReplicaTransformerEncoderLayer, ReplicaTransformerEncoder
from model.predictor.config import PolicyConfig
from model.modules.visual_modules import ImageEncoder, ImageDecoder, LanguageEncoder


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """모델의 예측값과 정답 값을 받아 전체 Loss를 계산합니다."""
    losses = {}
    weights = {
        'forward_state_loss': 1.0, 'backward_state_loss': 1.0,
        'goal_image_loss': 1.0, 'progress_loss': 0.5
    }

    # For goal image, resize target to match prediction if necessary
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        pred_img = predictions['predicted_goal_images']
        # Goal images from dataset are (B, 2, 3, H, W), we take the front view for loss
        true_img = targets['goal_images'][:, 0]

        # Predicted image can be 6-channel if decoder is designed that way
        # Let's adapt to what the decoder outputs. The current decoder outputs 3 channels.
        # We need to ensure the decoder's output channel matches one view.
        # Let's assume the decoder output is (B, 3, H, W) for the front view
        if pred_img.shape[1] == 6:  # If decoder outputs 6 channels
            pred_img = pred_img[:, :3]  # Take the first 3 for front view

        if pred_img.shape[2:] != true_img.shape[2:]:
            true_img = F.interpolate(
                true_img, size=pred_img.shape[2:], mode='bilinear', align_corners=False)

        losses['goal_image_loss'] = F.mse_loss(pred_img, true_img)

    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        losses['forward_state_loss'] = F.l1_loss(
            predictions['predicted_forward_states'], targets['forward_states'])

    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])

    if 'predicted_progress' in predictions and 'normalized_timestep' in targets:
        predicted = predictions['predicted_progress'].squeeze(-1)
        target = targets['normalized_timestep']
        losses['progress_loss'] = F.mse_loss(predicted, target)

    total_loss = torch.tensor(0.0, device=next(
        iter(predictions.values())).device)
    for name, loss in losses.items():
        if name in weights and loss is not None:
            total_loss += loss * weights.get(name, 1.0)

    return total_loss


class HierarchicalAutoregressivePolicy(nn.Module):
    def __init__(self, config: PolicyConfig, **kwargs):
        super().__init__()
        self.config = config
        h_cfg = config.hierarchical_transformer
        v_cfg = config.vision_encoder
        l_cfg = config.language_encoder

        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)

        self.arm_state_encoder = nn.Linear(7, h_cfg.hidden_dim)
        self.gripper_state_encoder = nn.Linear(1, h_cfg.hidden_dim)
        self.state_projector = nn.Linear(
            h_cfg.hidden_dim * 2, h_cfg.hidden_dim)

        self.lang_projection = nn.Linear(
            l_cfg.projection_dim, h_cfg.hidden_dim)
        self.image_feature_projector = nn.Linear(
            v_cfg.image_latent_dim * 2, h_cfg.hidden_dim)

        self.goal_query = nn.Parameter(torch.randn(
            1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.bwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_bwd_tokens, h_cfg.hidden_dim))
        self.fwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_fwd_tokens, h_cfg.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads, dim_feedforward=h_cfg.hidden_dim*4,
            dropout=h_cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.multi_modal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=h_cfg.num_layers, norm=nn.LayerNorm(h_cfg.hidden_dim))

        self.goal_head = nn.Linear(h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.bwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.backward_steps * h_cfg.state_dim)
        self.fwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.forward_steps * h_cfg.state_dim)

    def encode(self, initial_images, initial_states, language_instruction):
        """Encodes history into memory."""
        batch_size = initial_images.shape[0]
        n_obs_steps = initial_images.shape[1]

        # --- [FIXED] Unpack configs at the beginning of the method for local access ---
        v_cfg = self.config.vision_encoder
        h_cfg = self.config.hierarchical_transformer

        # 1. Image Processing
        images_flat = initial_images.flatten(0, 1)
        front_images = images_flat[:, 0]
        wrist_images = images_flat[:, 1]

        front_embeds = self.image_encoder(front_images)
        wrist_embeds = self.image_encoder(wrist_images)

        combined_image_feats = torch.cat([front_embeds, wrist_embeds], dim=-1)
        img_embeds_proj = self.image_feature_projector(combined_image_feats)
        img_embeds = img_embeds_proj.view(batch_size, n_obs_steps, -1)

        # 2. State Processing
        arm_states = initial_states[..., :7]
        gripper_states = initial_states[..., 7:]

        arm_embeds = self.arm_state_encoder(arm_states)
        gripper_embeds = self.gripper_state_encoder(gripper_states)
        combined_state_feats = torch.cat([arm_embeds, gripper_embeds], dim=-1)
        state_embeds = self.state_projector(combined_state_feats)

        # 3. Language Processing
        lang_embed = self.language_encoder(language_instruction).unsqueeze(1)
        lang_embed_proj = self.lang_projection(lang_embed)

        # Concat tokens
        seq = torch.cat([
            lang_embed_proj, state_embeds, img_embeds,
            self.goal_query.expand(batch_size, -1, -1),
            self.bwd_query.expand(batch_size, -1, -1),
            self.fwd_query.expand(batch_size, -1, -1)
        ], dim=1)

        # Build attention mask
        seq_len = seq.size(1)
        mask = torch.ones(seq_len, seq_len, device=seq.device,
                          dtype=torch.bool).triu(1)

        hist_len = lang_embed_proj.size(
            1) + state_embeds.size(1) + img_embeds.size(1)
        goal_start = hist_len
        bwd_start = goal_start + h_cfg.num_goal_tokens
        fwd_start = bwd_start + h_cfg.num_bwd_tokens

        mask[goal_start:, :hist_len] = False
        mask[bwd_start:, goal_start:bwd_start] = False
        mask[fwd_start:, bwd_start:fwd_start] = False

        out = self.multi_modal_encoder(seq, mask=mask)

        # Slice outputs
        goal_out = out[:, goal_start: goal_start +
                       h_cfg.num_goal_tokens].mean(dim=1)
        bwd_out = out[:, bwd_start: bwd_start +
                      h_cfg.num_bwd_tokens].mean(dim=1)
        fwd_out = out[:, fwd_start: fwd_start +
                      h_cfg.num_fwd_tokens].mean(dim=1)

        return goal_out, bwd_out, fwd_out

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        """Efficiently processes all predictions in a single pass for training."""
        h_cfg = self.config.hierarchical_transformer
        goal_h, bwd_h, fwd_h = self.encode(
            initial_images, initial_states, language_instruction)

        preds = {
            "predicted_goal_latents": self.goal_head(goal_h),
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, h_cfg.state_dim),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, h_cfg.state_dim),
        }
        preds["predicted_goal_images"] = self.image_decoder(
            preds["predicted_goal_latents"])
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        """Generates predictions for inference."""
        return self.forward(initial_images, initial_states, language_instruction)
