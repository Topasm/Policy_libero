# model/backbone/transformer_backbone.py
import torch
from torch import nn, Tensor
from typing import Optional, Callable
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(
                x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


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

        for layer in self.layers:
            output = layer(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
