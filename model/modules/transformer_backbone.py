# model/backbone/transformer_backbone.py
import torch
from torch import nn, Tensor
from typing import Optional, Callable

# RMSNorm은 custom_transformer.py에서 가져오거나 여기에 복사할 수 있습니다.
# 여기서는 의존성을 줄이기 위해 직접 정의합니다.


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dims**-0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        return x / (norm_x * self.scale + self.eps) * self.gamma


class FeedForward(nn.Module):
    """ Standard FeedForward network used in Transformer blocks. """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = nn.functional.relu):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CustomDecoderLayer(nn.Module):
    """
    A custom Transformer Decoder-style layer that uses pre-LayerNorm.
    It performs self-attention only.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 activation: Callable[[Tensor], Tensor], batch_first: bool = True):
        super().__init__()
        if not batch_first:
            raise ValueError(
                "This custom layer only supports batch_first=True")

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # Pre-Norm structure
        x = src
        # Self-attention block
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(
            x), self.norm1(x), attn_mask=src_mask)
        x = x + self.dropout1(attn_out)
        # Feed-forward block
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


class CustomDecoder(nn.Module):
    """ A stack of N custom decoder layers. """

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
