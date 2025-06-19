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
from collections import deque

from config.config import PolicyConfig
from model.modules.visual_modules import ImageEncoder, ImageDecoder, LanguageEncoder
from model.modules.transformer_backbone import CustomDecoder, CustomDecoderLayer, RMSNorm
from lerobot.common.policies.normalize import Normalize, Unnormalize


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    losses = {}
    weights = {
        'forward_action_loss_arm': 1.0,
        'forward_action_loss_gripper': 0.01,
        'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,
        'goal_progress_loss': 0.3,
        'backward_progress_loss': 0.3,
        'forward_progress_loss': 0.3,
    }

    # Goal image loss (both front and wrist)
    if 'predicted_goal_image_front' in predictions and 'goal_images' in targets:
        pred_front = predictions['predicted_goal_image_front']
        pred_wrist = predictions['predicted_goal_image_wrist']
        true_front = targets['goal_images'][:, 0]
        true_wrist = targets['goal_images'][:, 1]

        # Handle size mismatches with interpolation
        if pred_front.shape[2:] != true_front.shape[2:]:
            true_front = F.interpolate(
                true_front, size=pred_front.shape[2:], mode='bilinear', align_corners=False)

        if pred_wrist.shape[2:] != true_wrist.shape[2:]:
            true_wrist = F.interpolate(
                true_wrist, size=pred_wrist.shape[2:], mode='bilinear', align_corners=False)

        loss_front = F.mse_loss(pred_front, true_front)
        loss_wrist = F.mse_loss(pred_wrist, true_wrist)
        losses['goal_image_loss'] = loss_front + loss_wrist

    # Backward state loss
    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])

    # Forward action loss
    if 'predicted_forward_actions' in predictions and 'action' in targets:
        pred_actions = predictions['predicted_forward_actions']
        true_actions = targets['action'][:, :pred_actions.shape[1]]
        pad_mask = targets['action_is_pad'][:, :pred_actions.shape[1]]

        pred_arm, pred_gripper_logit = pred_actions[...,
                                                    :6], pred_actions[..., 6:]
        true_arm, true_gripper = true_actions[..., :6], true_actions[..., 6:]

        loss_arm_per_element = F.l1_loss(pred_arm, true_arm, reduction='none')
        true_gripper_bce = (true_gripper + 1.0) / 2.0
        loss_gripper_per_element = F.binary_cross_entropy_with_logits(
            pred_gripper_logit, true_gripper_bce, reduction='none'
        )

        mask = ~pad_mask
        loss_arm_per_element *= mask.unsqueeze(-1)
        loss_gripper_per_element *= mask.unsqueeze(-1)

        num_valid_elements = mask.sum()
        losses['forward_action_loss_arm'] = loss_arm_per_element.sum() / \
            (num_valid_elements * 6 + 1e-8)
        losses['forward_action_loss_gripper'] = loss_gripper_per_element.sum(
        ) / (num_valid_elements * 1 + 1e-8)

    # Progress prediction losses for each query
    if 'normalized_timestep' in targets:
        target_progress = targets['normalized_timestep']

        if 'goal_predicted_progress' in predictions:
            predicted_goal_progress = predictions['goal_predicted_progress'].squeeze(
                -1)
            losses['goal_progress_loss'] = F.mse_loss(
                predicted_goal_progress, target_progress)

        if 'backward_predicted_progress' in predictions:
            predicted_bwd_progress = predictions['backward_predicted_progress'].squeeze(
                -1)
            losses['backward_progress_loss'] = F.mse_loss(
                predicted_bwd_progress, target_progress)

        if 'forward_predicted_progress' in predictions:
            predicted_fwd_progress = predictions['forward_predicted_progress'].squeeze(
                -1)
            losses['forward_progress_loss'] = F.mse_loss(
                predicted_fwd_progress, target_progress)

    # Calculate total loss
    total_loss = torch.tensor(0.0, device=next(
        iter(predictions.values())).device)
    for name, loss in losses.items():
        if name in weights and loss is not None and loss.numel() > 0:
            total_loss += loss * weights.get(name, 1.0)
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

        # --- Input Encoders ---
        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)

        # --- Projectors ---
        self.state_projection = nn.Linear(h_cfg.state_dim, h_cfg.hidden_dim)
        self.lang_projection = nn.Linear(l_cfg.embedding_dim, h_cfg.hidden_dim)
        vit_hidden_size = self.image_encoder.vit.config.hidden_size
        self.patch_token_projector = nn.Linear(
            vit_hidden_size, h_cfg.hidden_dim)
        self.cls_token_projector = nn.Linear(vit_hidden_size, h_cfg.hidden_dim)

        # --- Queries for Hierarchical, End-to-End Prediction (REMOVED progress_query) ---
        self.goal_query = nn.Parameter(torch.randn(
            1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.bwd_query = nn.Parameter(torch.randn(
            1, h_cfg.backward_steps, h_cfg.hidden_dim))
        self.action_query = nn.Parameter(torch.randn(
            1, d_cfg.n_action_steps, h_cfg.hidden_dim))

        # --- Positional Embedding ---
        num_lang_tokens = 1
        num_state_tokens = d_cfg.n_obs_steps
        num_image_tokens = d_cfg.n_obs_steps * \
            (v_cfg.num_query_per_image * len(d_cfg.image_keys) + 2)
        # REMOVED progress query from count
        num_query_tokens = h_cfg.num_goal_tokens + \
            h_cfg.backward_steps + d_cfg.n_action_steps
        max_len = num_lang_tokens + num_state_tokens + \
            num_image_tokens + num_query_tokens + 16
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, h_cfg.hidden_dim))

        # --- Custom Decoder-Only Backbone ---
        decoder_layer = CustomDecoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads,
            dim_feedforward=h_cfg.hidden_dim * 4, dropout=h_cfg.dropout,
            activation=F.gelu, batch_first=True)
        self.multi_modal_backbone = CustomDecoder(
            decoder_layer, num_layers=h_cfg.num_layers, norm=RMSNorm(h_cfg.hidden_dim))

        # --- Prediction Heads ---
        mlp_hidden_dim = h_cfg.hidden_dim // 2

        # Shared progress head for all queries
        self.progress_head = nn.Sequential(
            nn.Linear(h_cfg.hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )

        # Goal prediction heads
        self.goal_head_front = nn.Linear(
            h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.goal_head_wrist = nn.Linear(
            h_cfg.hidden_dim, v_cfg.image_latent_dim)

        self.bwd_head = nn.Sequential(
            nn.Linear(h_cfg.hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, h_cfg.state_dim))

        self.action_decoder_body = nn.Sequential(
            nn.Linear(h_cfg.hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_decoder_arm_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, 6),
            nn.Tanh()
        )
        self.action_decoder_gripper_head = nn.Linear(
            mlp_hidden_dim, 1)  # Outputs logits

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        batch_size, n_obs_steps = initial_images.shape[:2]
        h_cfg = self.config.hierarchical_transformer
        d_cfg = self.config.data

        # 1. Embed all inputs
        images_flat = initial_images.flatten(0, 1)
        front_images, wrist_images = images_flat[:, 0], images_flat[:, 1]
        front_patch_tokens, front_cls = self.image_encoder(front_images)
        wrist_patch_tokens, wrist_cls = self.image_encoder(wrist_images)
        front_patch_embeds = self.patch_token_projector(front_patch_tokens)
        wrist_patch_embeds = self.patch_token_projector(wrist_patch_tokens)
        front_cls_embed = self.cls_token_projector(front_cls)
        wrist_cls_embed = self.cls_token_projector(wrist_cls)
        image_tokens = torch.cat(
            [front_patch_embeds, wrist_patch_embeds, front_cls_embed, wrist_cls_embed], dim=1)
        img_embeds = image_tokens.view(
            batch_size, n_obs_steps * image_tokens.shape[1], -1)
        state_embeds = self.state_projection(initial_states)
        lang_embed = self.language_encoder(language_instruction).unsqueeze(1)
        lang_embed_proj = self.lang_projection(lang_embed)

        # 2. Assemble the full sequence (REMOVED progress_query)
        history_embeds = torch.cat(
            [lang_embed_proj, state_embeds, img_embeds], dim=1)
        query_embeds = torch.cat([
            self.goal_query.expand(batch_size, -1, -1),
            self.bwd_query.expand(batch_size, -1, -1),
            self.action_query.expand(batch_size, -1, -1)
        ], dim=1)
        seq = torch.cat([history_embeds, query_embeds], dim=1)
        seq = seq + self.pos_embed[:, :seq.shape[1], :]

        # 3. Create the custom attention mask (UPDATED indices)
        seq_len = seq.shape[1]
        mask = torch.ones(seq_len, seq_len, device=seq.device,
                          dtype=torch.bool).triu(1)

        hist_len = history_embeds.size(1)
        goal_start = hist_len
        bwd_start = goal_start + h_cfg.num_goal_tokens
        action_start = bwd_start + h_cfg.backward_steps

        # Updated mask logic without progress query
        mask[goal_start:, :hist_len] = False
        mask[bwd_start:, goal_start:bwd_start] = False
        mask[action_start:, bwd_start:bwd_start] = False  # Action also sees Bwd
        # Action also sees Goal
        mask[action_start:, goal_start:goal_start] = False

        # 4. Pass through the backbone
        out = self.multi_modal_backbone(seq, mask=mask)

        # 5. Slice outputs (UPDATED indices)
        goal_h = out[:, goal_start:bwd_start].mean(dim=1)
        bwd_h_sequence = out[:, bwd_start:action_start]
        action_h_sequence = out[:, action_start:]

        # --- Generate final predictions with progress for each query ---
        pred_bwd_states = self.bwd_head(bwd_h_sequence)
        shared_features = self.action_decoder_body(action_h_sequence)
        pred_arm = self.action_decoder_arm_head(shared_features)
        pred_gripper_logit = self.action_decoder_gripper_head(shared_features)
        pred_fwd_actions = torch.cat([pred_arm, pred_gripper_logit], dim=-1)
        pred_latent_front = self.goal_head_front(goal_h)
        pred_latent_wrist = self.goal_head_wrist(goal_h)
        pred_img_front, pred_img_wrist = self.image_decoder(
            pred_latent_front, pred_latent_wrist)

        # Progress predictions for each query
        goal_progress = self.progress_head(goal_h)
        bwd_progress = self.progress_head(bwd_h_sequence.mean(dim=1))
        action_progress = self.progress_head(action_h_sequence.mean(dim=1))

        preds = {
            "predicted_backward_states": pred_bwd_states,
            "predicted_forward_actions": pred_fwd_actions,
            "predicted_goal_image_front": pred_img_front,
            "predicted_goal_image_wrist": pred_img_wrist,
            "goal_predicted_progress": goal_progress,
            "backward_predicted_progress": bwd_progress,
            "forward_predicted_progress": action_progress,
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        # This would need a proper inference-time implementation now
        return self.forward(initial_images, initial_states, language_instruction)
