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
from typing import Dict

from config.config import PolicyConfig
from model.modules.visual_modules import ImageEncoder, ImageDecoder, LanguageEncoder
from model.modules.transformer_backbone import CustomDecoder, CustomDecoderLayer, RMSNorm


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    losses = {}
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,
        'progress_loss': 0.5  # 진행률 손실 가중치
    }

    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        pred_img = predictions['predicted_goal_images']
        true_img_front = targets['goal_images'][:, 0]
        if pred_img.shape[2:] != true_img_front.shape[2:]:
            true_img_front = F.interpolate(
                true_img_front, size=pred_img.shape[2:], mode='bilinear', align_corners=False)
        losses['goal_image_loss'] = F.mse_loss(pred_img, true_img_front)

    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        losses['forward_state_loss'] = F.l1_loss(
            predictions['predicted_forward_states'], targets['forward_states'])

    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])

    # [MODIFIED] Ensure progress loss is calculated
    if 'predicted_progress' in predictions and 'normalized_timestep' in targets:
        predicted = predictions['predicted_progress'].squeeze(-1)
        target = targets['normalized_timestep']
        losses['progress_loss'] = F.mse_loss(predicted, target)

    total_loss = torch.tensor(0.0, device=next(
        iter(predictions.values())).device)

    # Calculate weighted sum of losses
    for name, loss in losses.items():
        if name in weights and loss is not None:
            total_loss += loss * weights.get(name, 1.0)

    # Add the total loss to the dictionary
    losses['total_loss'] = total_loss

    return losses


class HierarchicalAutoregressivePolicy(nn.Module):
    def __init__(self, config: PolicyConfig, **kwargs):
        super().__init__()
        self.config = config
        h_cfg = config.hierarchical_transformer
        v_cfg = config.vision_encoder
        l_cfg = config.language_encoder
        d_cfg = config.data

        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)
        self.state_projection = nn.Linear(h_cfg.state_dim, h_cfg.hidden_dim)
        self.lang_projection = nn.Linear(
            l_cfg.projection_dim, h_cfg.hidden_dim)
        if v_cfg.image_latent_dim != h_cfg.hidden_dim:
            self.image_token_projector = nn.Linear(
                v_cfg.image_latent_dim, h_cfg.hidden_dim)
        else:
            self.image_token_projector = nn.Identity()

        # [MODIFIED] Add a new query token for progress prediction
        self.progress_query = nn.Parameter(torch.randn(1, 1, h_cfg.hidden_dim))
        self.goal_query = nn.Parameter(torch.randn(
            1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.bwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_bwd_tokens, h_cfg.hidden_dim))
        self.fwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_fwd_tokens, h_cfg.hidden_dim))

        # [MODIFIED] Update max sequence length calculation for the new query
        num_lang_tokens = 1
        num_state_tokens = d_cfg.n_obs_steps
        num_image_tokens = d_cfg.n_obs_steps * \
            v_cfg.num_latents_per_image * len(d_cfg.image_keys)
        num_query_tokens = 1 + h_cfg.num_goal_tokens + \
            h_cfg.num_bwd_tokens + h_cfg.num_fwd_tokens  # Now 4 queries
        max_len = num_lang_tokens + num_state_tokens + \
            num_image_tokens + num_query_tokens + 16
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, h_cfg.hidden_dim))

        decoder_layer = CustomDecoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads,
            dim_feedforward=h_cfg.hidden_dim * 4, dropout=h_cfg.dropout,
            activation=F.gelu, batch_first=True
        )
        self.multi_modal_backbone = CustomDecoder(
            decoder_layer, num_layers=h_cfg.num_layers, norm=RMSNorm(
                h_cfg.hidden_dim)
        )

        self.goal_head = nn.Linear(h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.bwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.backward_steps * h_cfg.state_dim)
        self.fwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.forward_steps * h_cfg.state_dim)
        # [MODIFIED] Add a new head for progress prediction
        self.progress_head = nn.Linear(h_cfg.hidden_dim, 1)

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        batch_size = initial_images.shape[0]
        n_obs_steps = initial_images.shape[1]
        h_cfg = self.config.hierarchical_transformer

        # 1. Input Embedding
        images_flat = initial_images.flatten(0, 1)
        front_images, wrist_images = images_flat[:, 0], images_flat[:, 1]
        front_tokens, wrist_tokens = self.image_encoder(
            front_images), self.image_encoder(wrist_images)
        front_tokens, wrist_tokens = self.image_token_projector(
            front_tokens), self.image_token_projector(wrist_tokens)
        image_tokens = torch.cat([front_tokens, wrist_tokens], dim=1)
        img_embeds = image_tokens.view(
            batch_size, n_obs_steps * image_tokens.shape[1], -1)
        state_embeds = self.state_projection(initial_states)
        lang_embed = self.language_encoder(language_instruction).unsqueeze(1)
        lang_embed_proj = self.lang_projection(lang_embed)

        # 2. Assemble Full Sequence (History + Queries)
        history_embeds = torch.cat(
            [lang_embed_proj, state_embeds, img_embeds], dim=1)

        # [MODIFIED] Add progress_query to the sequence, before the goal_query
        query_embeds = torch.cat([
            self.progress_query.expand(batch_size, -1, -1),
            self.goal_query.expand(batch_size, -1, -1),
            self.bwd_query.expand(batch_size, -1, -1),
            self.fwd_query.expand(batch_size, -1, -1)
        ], dim=1)

        seq = torch.cat([history_embeds, query_embeds], dim=1)
        seq = seq + self.pos_embed[:, :seq.shape[1], :]

        # 3. Create Custom Attention Mask for the new hierarchy
        seq_len = seq.shape[1]
        custom_mask_bool = torch.ones(
            seq_len, seq_len, device=seq.device, dtype=torch.bool).triu(1)

        hist_len = history_embeds.size(1)
        # Define start indices for all queries
        prog_start = hist_len
        goal_start = prog_start + 1  # num_progress_tokens is 1
        bwd_start = goal_start + h_cfg.num_goal_tokens
        fwd_start = bwd_start + h_cfg.num_bwd_tokens

        # [MODIFIED] Update mask logic for the new hierarchy
        # All queries see history
        custom_mask_bool[prog_start:, :hist_len] = False
        # Goal, Bwd, Fwd see Progress
        custom_mask_bool[goal_start:, prog_start:goal_start] = False
        # Bwd, Fwd see Goal
        custom_mask_bool[bwd_start:, goal_start:bwd_start] = False
        # Fwd sees Bwd
        custom_mask_bool[fwd_start:, bwd_start:fwd_start] = False

        # 4. Run CustomDecoder Backbone
        out = self.multi_modal_backbone(seq, mask=custom_mask_bool)

        # 5. Slice outputs and make final predictions
        prog_h = out[:, prog_start: prog_start + 1].mean(dim=1)
        goal_h = out[:, goal_start: goal_start +
                     h_cfg.num_goal_tokens].mean(dim=1)
        bwd_h = out[:, bwd_start: bwd_start + h_cfg.num_bwd_tokens].mean(dim=1)
        fwd_h = out[:, fwd_start: fwd_start + h_cfg.num_fwd_tokens].mean(dim=1)

        preds = {
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, h_cfg.state_dim),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, h_cfg.state_dim),
            "predicted_goal_images": self.image_decoder(self.goal_head(goal_h)),
            # Add progress prediction
            "predicted_progress": torch.sigmoid(self.progress_head(prog_h)),
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        # Generate now also produces progress, but it's mainly for loss calculation during training
        return self.forward(initial_images, initial_states, language_instruction)
