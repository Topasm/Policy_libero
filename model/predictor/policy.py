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
        'action_loss_arm': 1.0,
        'action_loss_gripper': 0.01,  # Gripper loss weight inspired by Seer
        'goal_image_loss': 1.0,
        'progress_loss': 0.5
    }

    # 1. Goal Image Reconstruction Loss
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        pred_img = predictions['predicted_goal_images']
        true_img_front = targets['goal_images'][:, 0]
        if pred_img.shape[2:] != true_img_front.shape[2:]:
            true_img_front = F.interpolate(
                true_img_front, size=pred_img.shape[2:], mode='bilinear', align_corners=False)
        losses['goal_image_loss'] = F.mse_loss(pred_img, true_img_front)

    # 2. Action Prediction Loss (L1 for arm, BCE for gripper)
    if 'predicted_actions' in predictions and 'action' in targets:
        pred_actions = predictions['predicted_actions']
        true_actions = targets['action']
        pred_arm, pred_gripper_logit = pred_actions[...,
                                                    :6], pred_actions[..., 6:]
        true_arm, true_gripper = true_actions[..., :6], true_actions[..., 6:]
        losses['action_loss_arm'] = F.l1_loss(pred_arm, true_arm)
        true_gripper_bce = (true_gripper + 1.0) / 2.0
        losses['action_loss_gripper'] = F.binary_cross_entropy_with_logits(
            pred_gripper_logit, true_gripper_bce)

    # 3. Progress Prediction Loss
    if 'predicted_progress' in predictions and 'normalized_timestep' in targets:
        predicted = predictions['predicted_progress'].squeeze(-1)
        target = targets['normalized_timestep']
        losses['progress_loss'] = F.mse_loss(predicted, target)

    total_loss = torch.tensor(0.0, device=next(
        iter(predictions.values())).device)
    for name, loss in losses.items():
        if name in weights and loss is not None:
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

        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)
        self.state_projection = nn.Linear(h_cfg.state_dim, h_cfg.hidden_dim)
        self.lang_projection = nn.Linear(l_cfg.embedding_dim, h_cfg.hidden_dim)

        vit_hidden_size = self.image_encoder.vit.config.hidden_size
        self.patch_token_projector = nn.Linear(
            vit_hidden_size, h_cfg.hidden_dim)
        self.cls_token_projector = nn.Linear(vit_hidden_size, h_cfg.hidden_dim)

        # [MODIFIED] Replaced state planning queries with an action query
        self.progress_query = nn.Parameter(torch.randn(1, 1, h_cfg.hidden_dim))
        self.goal_query = nn.Parameter(torch.randn(
            1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.action_query = nn.Parameter(torch.randn(
            1, d_cfg.n_action_steps, h_cfg.hidden_dim))

        # [MODIFIED] Update positional embedding size
        num_lang_tokens = 1
        num_state_tokens = d_cfg.n_obs_steps
        num_image_tokens = d_cfg.n_obs_steps * \
            (v_cfg.num_query_per_image * len(d_cfg.image_keys) + 2)
        num_query_tokens = 1 + h_cfg.num_goal_tokens + d_cfg.n_action_steps
        max_len = num_lang_tokens + num_state_tokens + \
            num_image_tokens + num_query_tokens + 16
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, h_cfg.hidden_dim))

        decoder_layer = CustomDecoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads,
            dim_feedforward=h_cfg.hidden_dim * 4, dropout=h_cfg.dropout,
            activation=F.gelu, batch_first=True)
        self.multi_modal_backbone = CustomDecoder(
            decoder_layer, num_layers=h_cfg.num_layers, norm=RMSNorm(h_cfg.hidden_dim))

        # [MODIFIED] Action decoder head
        mlp_hidden_dim = h_cfg.hidden_dim // 2
        self.goal_head = nn.Linear(h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.progress_head = nn.Linear(h_cfg.hidden_dim, 1)
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

        # 1. Input Embedding
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

        # 2. Assemble Full Sequence
        history_embeds = torch.cat(
            [lang_embed_proj, state_embeds, img_embeds], dim=1)
        query_embeds = torch.cat([
            self.progress_query.expand(batch_size, -1, -1),
            self.goal_query.expand(batch_size, -1, -1),
            self.action_query.expand(batch_size, -1, -1),
        ], dim=1)
        seq = torch.cat([history_embeds, query_embeds], dim=1)
        seq = seq + self.pos_embed[:, :seq.shape[1], :]

        # 3. Create Custom Attention Mask for new hierarchy: History -> Progress -> Goal -> Action
        seq_len = seq.shape[1]
        hist_len = history_embeds.size(1)
        prog_start = hist_len
        goal_start = prog_start + 1
        action_start = goal_start + h_cfg.num_goal_tokens

        custom_mask_bool = torch.ones(
            seq_len, seq_len, device=seq.device, dtype=torch.bool).triu(1)
        # All queries see history
        custom_mask_bool[prog_start:, :hist_len] = False
        # Goal & Action see Progress
        custom_mask_bool[goal_start:, prog_start:goal_start] = False
        custom_mask_bool[action_start:,
                         goal_start:action_start] = False  # Action sees Goal

        # 4. Run Backbone
        out = self.multi_modal_backbone(seq, mask=custom_mask_bool)

        # 5. Slice outputs
        prog_h = out[:, prog_start:goal_start].mean(dim=1)
        goal_h = out[:, goal_start:action_start].mean(dim=1)
        action_h_sequence = out[:, action_start:action_start +
                                self.action_query.shape[1]]

        # 6. Final Predictions: Goal Image and Actions
        pred_latent = self.goal_head(goal_h)
        pred_img = self.image_decoder(pred_latent)
        shared_features = self.action_decoder_body(action_h_sequence)
        pred_arm = self.action_decoder_arm_head(shared_features)
        pred_gripper_logit = self.action_decoder_gripper_head(shared_features)
        pred_actions = torch.cat([pred_arm, pred_gripper_logit], dim=-1)

        preds = {
            "predicted_actions": pred_actions,
            "predicted_goal_images": pred_img,
            "predicted_progress": torch.sigmoid(self.progress_head(prog_h)),
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        return self.forward(initial_images, initial_states, language_instruction)
