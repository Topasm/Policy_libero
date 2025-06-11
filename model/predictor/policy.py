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
    losses = {}
    weights = {
        'forward_state_loss': 1.0, 'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,  # This loss will now be the sum of front and wrist image losses
    }

    # [MODIFIED] Calculate loss for both front and wrist goal images
    if 'predicted_goal_image_front' in predictions and 'goal_images' in targets:
        # Predicted images
        pred_front = predictions['predicted_goal_image_front']
        pred_wrist = predictions['predicted_goal_image_wrist']

        # Ground truth images from the dataset (B, 2, 3, H, W)
        true_front = targets['goal_images'][:, 0]
        true_wrist = targets['goal_images'][:, 1]

        # Calculate separate losses and add them up
        loss_front = F.mse_loss(pred_front, true_front)
        loss_wrist = F.mse_loss(pred_wrist, true_wrist)
        losses['goal_image_loss'] = loss_front + loss_wrist

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

        # --- [MODIFIED] Separate State Encoders for Arm (6D) and Gripper (2D) ---
        self.arm_state_encoder = nn.Linear(6, h_cfg.hidden_dim)
        self.gripper_state_encoder = nn.Linear(2, h_cfg.hidden_dim)
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
        # [MODIFIED] Create two separate prediction heads for goal latents
        self.goal_head_front = nn.Linear(
            h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.goal_head_wrist = nn.Linear(
            h_cfg.hidden_dim, v_cfg.image_latent_dim)
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
        arm_states = initial_states[..., :6]
        gripper_states = initial_states[..., 6:]

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
        """Processes all predictions, now including separate goal images."""
        h_cfg = self.config.hierarchical_transformer
        goal_h, bwd_h, fwd_h = self.encode(
            initial_images, initial_states, language_instruction)

        # [MODIFIED] Predict two separate latents and decode them into two images
        pred_latent_front = self.goal_head_front(goal_h)
        pred_latent_wrist = self.goal_head_wrist(goal_h)

        pred_img_front, pred_img_wrist = self.image_decoder(
            pred_latent_front, pred_latent_wrist)

        preds = {
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, h_cfg.state_dim),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, h_cfg.state_dim),
            "predicted_goal_image_front": pred_img_front,
            "predicted_goal_image_wrist": pred_img_wrist,
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        """Generates predictions for inference."""
        return self.forward(initial_images, initial_states, language_instruction)
