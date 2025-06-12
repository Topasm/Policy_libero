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


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = {}
    weights = {
        'forward_state_loss': 1.0, 'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,
    }

    # [MODIFIED] Calculate loss only for the single predicted goal image vs. the front ground truth image.
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        pred_img = predictions['predicted_goal_images']
        # Ground truth `goal_images` has shape (B, 2, 3, H, W). We take the front view (index 0).
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
        d_cfg = config.data

        # --- 입력 인코더 ---
        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)

        for param in self.image_encoder.vit.parameters():
            param.requires_grad = False
        print("Froze ImageEncoder (ViT) backbone.")

        # [MODIFIED] Reverted to a single, unified state encoder
        self.state_projection = nn.Linear(h_cfg.state_dim, h_cfg.hidden_dim)

        self.lang_projection = nn.Linear(
            l_cfg.projection_dim, h_cfg.hidden_dim)
        if v_cfg.image_latent_dim != h_cfg.hidden_dim:
            self.image_token_projector = nn.Linear(
                v_cfg.image_latent_dim, h_cfg.hidden_dim)
        else:
            self.image_token_projector = nn.Identity()

        self.goal_query = nn.Parameter(torch.randn(
            1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.bwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_bwd_tokens, h_cfg.hidden_dim))
        self.fwd_query = nn.Parameter(torch.randn(
            1, h_cfg.num_fwd_tokens, h_cfg.hidden_dim))
        num_lang_tokens = 1
        num_state_tokens = d_cfg.n_obs_steps
        num_image_tokens = d_cfg.n_obs_steps * \
            v_cfg.num_latents_per_image * len(d_cfg.image_keys)
        num_query_tokens = h_cfg.num_goal_tokens + \
            h_cfg.num_bwd_tokens + h_cfg.num_fwd_tokens
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

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        """
        [MODIFIED] 모든 로직을 forward 함수 안에서 처리하는 통합 디코더-온리 구조.
        """
        batch_size = initial_images.shape[0]
        n_obs_steps = initial_images.shape[1]
        h_cfg = self.config.hierarchical_transformer

        # --- 1. 입력 임베딩 ---
        images_flat = initial_images.flatten(0, 1)
        front_images, wrist_images = images_flat[:, 0], images_flat[:, 1]
        front_tokens, wrist_tokens = self.image_encoder(
            front_images), self.image_encoder(wrist_images)
        front_tokens, wrist_tokens = self.image_token_projector(
            front_tokens), self.image_token_projector(wrist_tokens)
        image_tokens = torch.cat([front_tokens, wrist_tokens], dim=1)
        img_embeds = image_tokens.view(
            batch_size, n_obs_steps * image_tokens.shape[1], -1)

        # [MODIFIED] State Processing using a single projection
        state_embeds = self.state_projection(initial_states)

        lang_embed = self.language_encoder(language_instruction).unsqueeze(1)
        lang_embed_proj = self.lang_projection(lang_embed)

        # --- 2. 전체 시퀀스 조립 및 위치 임베딩 추가 ---
        seq = torch.cat([
            lang_embed_proj, state_embeds, img_embeds,
            self.goal_query.expand(batch_size, -1, -1),
            self.bwd_query.expand(batch_size, -1, -1),
            self.fwd_query.expand(batch_size, -1, -1)
        ], dim=1)
        seq = seq + self.pos_embed[:, :seq.shape[1], :]

        # --- 3. 커스텀 어텐션 마스크 생성 ---
        seq_len = seq.shape[1]
        custom_mask_bool = torch.ones(
            seq_len, seq_len, device=seq.device, dtype=torch.bool).triu(1)

        hist_len = lang_embed_proj.size(
            1) + state_embeds.size(1) + img_embeds.size(1)
        goal_start = hist_len
        bwd_start = goal_start + h_cfg.num_goal_tokens
        fwd_start = bwd_start + h_cfg.num_bwd_tokens

        custom_mask_bool[goal_start:, :hist_len] = False
        custom_mask_bool[bwd_start:, goal_start:bwd_start] = False
        custom_mask_bool[fwd_start:, bwd_start:fwd_start] = False

        # --- 4. CustomDecoder 백본 실행 ---
        out = self.multi_modal_backbone(seq, mask=custom_mask_bool)

        # --- 5. 결과 슬라이싱 및 최종 예측 ---
        goal_h = out[:, goal_start: goal_start +
                     h_cfg.num_goal_tokens].mean(dim=1)
        bwd_h = out[:, bwd_start: bwd_start + h_cfg.num_bwd_tokens].mean(dim=1)
        fwd_h = out[:, fwd_start: fwd_start + h_cfg.num_fwd_tokens].mean(dim=1)

        pred_latent = self.goal_head(goal_h)
        pred_img = self.image_decoder(pred_latent)

        preds = {
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, h_cfg.state_dim),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, h_cfg.state_dim),
            "predicted_goal_images": pred_img,
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        return self.forward(initial_images, initial_states, language_instruction)
