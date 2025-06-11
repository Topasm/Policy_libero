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

from model.predictor.config import PolicyConfig
from model.modules.visual_modules import ImageEncoder, ImageDecoder, LanguageEncoder


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

        # --- 입력 인코더 부분 (Seer 스타일의 개선된 버전 유지) ---
        self.image_encoder = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)
        self.image_decoder = ImageDecoder(v_cfg)
        self.arm_state_encoder = nn.Linear(6, h_cfg.hidden_dim)
        self.gripper_state_encoder = nn.Linear(2, h_cfg.hidden_dim)
        self.state_projector = nn.Linear(
            h_cfg.hidden_dim * 2, h_cfg.hidden_dim)
        self.lang_projection = nn.Linear(
            l_cfg.projection_dim, h_cfg.hidden_dim)
        if v_cfg.image_latent_dim != h_cfg.hidden_dim:
            self.image_token_projector = nn.Linear(
                v_cfg.image_latent_dim, h_cfg.hidden_dim)
        else:
            self.image_token_projector = nn.Identity()

        # --- [MODIFIED] Transformer Backbone: Reverted to Encoder-Decoder structure ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads, dim_feedforward=h_cfg.hidden_dim*4,
            dropout=h_cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=h_cfg.num_layers, norm=nn.LayerNorm(h_cfg.hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads, dim_feedforward=h_cfg.hidden_dim*4,
            dropout=h_cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.prediction_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=h_cfg.num_layers, norm=nn.LayerNorm(h_cfg.hidden_dim))

        # --- 쿼리 토큰 및 위치 임베딩 (인코더-디코더 구조용) ---
        self.query_embedding = nn.Embedding(
            3, h_cfg.hidden_dim)  # 0:Goal, 1:Bwd, 2:Fwd
        # 입력 이력의 최대 길이를 계산하여 위치 임베딩 생성
        num_lang_tokens = 1
        num_state_tokens = config.data.n_obs_steps
        num_image_tokens = config.data.n_obs_steps * \
            config.vision_encoder.num_latents_per_image * \
            len(config.data.image_keys)
        max_history_len = num_lang_tokens + num_state_tokens + \
            num_image_tokens + 16  # Add buffer
        self.history_pos_embedding = nn.Parameter(
            torch.randn(1, max_history_len, h_cfg.hidden_dim))

        # --- 예측 헤드 ---
        self.goal_head = nn.Linear(h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.bwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.backward_steps * h_cfg.state_dim)
        self.fwd_head = nn.Linear(
            h_cfg.hidden_dim, h_cfg.forward_steps * h_cfg.state_dim)

    def encode(self, initial_images, initial_states, language_instruction):
        """과거 이력을 인코딩하여 memory 텐서를 생성합니다."""
        batch_size = initial_images.shape[0]
        n_obs_steps = initial_images.shape[1]

        # 1. 각 입력을 임베딩
        images_flat = initial_images.flatten(0, 1)
        front_images, wrist_images = images_flat[:, 0], images_flat[:, 1]
        front_tokens, wrist_tokens = self.image_encoder(
            front_images), self.image_encoder(wrist_images)
        front_tokens, wrist_tokens = self.image_token_projector(
            front_tokens), self.image_token_projector(wrist_tokens)
        image_tokens = torch.cat([front_tokens, wrist_tokens], dim=1)
        img_embeds = image_tokens.view(
            batch_size, n_obs_steps * image_tokens.shape[1], -1)

        arm_states, gripper_states = initial_states[...,
                                                    :6], initial_states[..., 6:]
        arm_embeds, gripper_embeds = self.arm_state_encoder(
            arm_states), self.gripper_state_encoder(gripper_states)
        state_embeds = self.state_projector(
            torch.cat([arm_embeds, gripper_embeds], dim=-1))

        lang_embed = self.language_encoder(language_instruction).unsqueeze(1)
        lang_embed_proj = self.lang_projection(lang_embed)

        # 2. 이력 시퀀스 결합 및 위치 임베딩 추가
        history_sequence = torch.cat(
            [lang_embed_proj, state_embeds, img_embeds], dim=1)
        seq_len = history_sequence.shape[1]
        history_sequence = history_sequence + \
            self.history_pos_embedding[:, :seq_len, :]

        # 3. TransformerEncoder를 통과시켜 memory 생성
        return self.context_encoder(history_sequence)

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        """학습을 위한 forward 함수. 인코더-디코더 구조."""
        device = initial_images.device
        batch_size = initial_images.shape[0]

        memory = self.encode(
            initial_images, initial_states, language_instruction)

        query_ids = torch.arange(3, device=device).unsqueeze(
            0).expand(batch_size, -1)
        query_embeds = self.query_embedding(query_ids)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            query_embeds.size(1), device=device)

        decoder_output = self.prediction_decoder(
            tgt=query_embeds, memory=memory, tgt_mask=tgt_mask)

        goal_h, bwd_h, fwd_h = decoder_output[:,
                                              0], decoder_output[:, 1], decoder_output[:, 2]

        pred_latent = self.goal_head(goal_h)
        pred_img = self.image_decoder(pred_latent)

        h_cfg = self.config.hierarchical_transformer
        preds = {
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, h_cfg.state_dim),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, h_cfg.state_dim),
            "predicted_goal_images": pred_img,
        }
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        return self.forward(initial_images, initial_states, language_instruction)
