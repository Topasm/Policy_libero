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

from model.modules.modules import SpatialSoftmax
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

    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        losses['forward_state_loss'] = F.l1_loss(
            predictions['predicted_forward_states'], targets['forward_states'])
    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'], targets['goal_images'])
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

# --- 2. 최종 정책 모델 ---


class HierarchicalAutoregressivePolicy(nn.Module):
    def __init__(self, config: PolicyConfig, **kwargs):
        super().__init__()
        self.config = config
        # unpack sub-configs
        h_cfg = config.hierarchical_transformer
        v_cfg = config.vision_encoder
        l_cfg = config.language_encoder
        d_cfg = config.data

        # --- module initialization ---
        self.image_encoder   = ImageEncoder(v_cfg)
        self.language_encoder = LanguageEncoder(l_cfg)  # <--- NEW
        self.image_decoder   = ImageDecoder(v_cfg)
        self.state_projection = nn.Linear(h_cfg.state_dim, h_cfg.hidden_dim)
        self.lang_projection  = nn.Linear(l_cfg.projection_dim, h_cfg.hidden_dim)  # <--- NEW

        # new learnable query tokens
        self.goal_query = nn.Parameter(torch.randn(1, h_cfg.num_goal_tokens, h_cfg.hidden_dim))
        self.bwd_query  = nn.Parameter(torch.randn(1, h_cfg.num_bwd_tokens, h_cfg.hidden_dim))
        self.fwd_query  = nn.Parameter(torch.randn(1, h_cfg.num_fwd_tokens, h_cfg.hidden_dim))

        # --- multi-modal encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_cfg.hidden_dim, nhead=h_cfg.num_heads, dim_feedforward=h_cfg.hidden_dim*4,
            dropout=h_cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.multi_modal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=h_cfg.num_layers, norm=nn.LayerNorm(h_cfg.hidden_dim))

        # --- output heads ---
        self.goal_head = nn.Linear(h_cfg.hidden_dim, v_cfg.image_latent_dim)
        self.bwd_head  = nn.Linear(h_cfg.hidden_dim, h_cfg.backward_steps * h_cfg.state_dim)
        self.fwd_head  = nn.Linear(h_cfg.hidden_dim, h_cfg.forward_steps  * h_cfg.state_dim)

    def encode(self, initial_images, initial_states, language_instruction):
        """과거 이력을 인코딩하여 memory를 생성합니다."""
        # embed each modality
        batch_size = initial_images.shape[0]
        img_embeds   = self.image_encoder(initial_images.flatten(0,1)).view(batch_size, d_cfg.n_obs_steps, -1)
        state_embeds = self.state_projection(initial_states)
        lang_embed   = self.language_encoder(language_instruction).unsqueeze(1)

        # concat: [lang, state_hist, img_hist, goal_q, bwd_q, fwd_q]
        seq = torch.cat([
            lang_embed,
            state_embeds,
            img_embeds,
            self.goal_query.expand(batch_size, -1, -1),
            self.bwd_query .expand(batch_size, -1, -1),
            self.fwd_query .expand(batch_size, -1, -1)
        ], dim=1)

        # build causal mask allowing cross-modal attends
        seq_len = seq.size(1)
        mask = torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool)
        mask.triu_(1)
        # allow bwd to see goal, and fwd to see goal+bwd
        goal_start = 1 + state_embeds.size(1) + img_embeds.size(1)
        bwd_start  = goal_start + h_cfg.num_goal_tokens
        fwd_start  = bwd_start + h_cfg.num_bwd_tokens
        mask[bwd_start:, goal_start:bwd_start] = False
        mask[fwd_start:, goal_start:fwd_start] = False

        out = self.multi_modal_encoder(seq, mask=mask)
        return out[:, goal_start], out[:, bwd_start], out[:, fwd_start]

    def forward(self, initial_images, initial_states, language_instruction, **kwargs):
        """
        학습을 위한 forward 함수. 한 번의 pass로 모든 예측을 효율적으로 처리합니다.
        """
        h_cfg = self.config.hierarchical_transformer
        goal_h, bwd_h, fwd_h = self.encode(initial_images, initial_states, language_instruction)
        preds = {
            "predicted_goal_latents": self.goal_head(goal_h),
            "predicted_backward_states": self.bwd_head(bwd_h).view(-1, h_cfg.backward_steps, -1),
            "predicted_forward_states": self.fwd_head(fwd_h).view(-1, h_cfg.forward_steps, -1),
        }
        preds["predicted_goal_images"] = self.image_decoder(preds["predicted_goal_latents"])
        return preds

    @torch.no_grad()
    def generate(self, initial_images, initial_states, language_instruction) -> Dict[str, torch.Tensor]:
        """추론을 위한 generate 함수. Goal -> Bwd -> Fwd 순서로 순차 생성합니다."""
        # reuse forward for generation
        return self.forward(initial_images, initial_states, language_instruction)
