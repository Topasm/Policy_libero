# model/backbone/transformer_backbone.py
import torch
from torch import nn, Tensor
from typing import Optional, Callable
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """ A wrapper for torch.nn.functional.rms_norm """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The learnable scaling parameter, gamma
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # [FIXED] Call F.rms_norm with the correct arguments
        # The normalized_shape is derived from the shape of the learnable weight.
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


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
    A custom Transformer Decoder-style layer that uses pre-normalization.
    Now correctly uses the RMSNorm wrapper.
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

        # These now correctly instantiate the fixed RMSNorm class
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(
            x), self.norm1(x), attn_mask=src_mask)
        x = x + self.dropout1(attn_out)
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
