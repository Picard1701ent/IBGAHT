import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

import config

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


def multi_head_attention_forward(query, key, value, n_head, attn_mask=None, **kwargs):
    batch, seq_len, dim = query.shape
    head_dim = dim // n_head
    assert dim % n_head == 0, "Dimensionality must be divisible by the number of heads"
    query = query.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

    attn_scores = torch.matmul(query, key.transpose(-2, -1))
    attn_scores = attn_scores / (head_dim**0.5)

    if attn_mask is not None:
        attn_scores = attn_scores * attn_mask.unsqueeze(1)
    unnorm_attn_weight = attn_scores.clone()

    attn_weights = torch.softmax(
        attn_scores, dim=-1
    )  # (batch, n_head, seq_len, seq_len)

    output = torch.matmul(attn_weights, value)  # (batch, n_head, seq_len, head_dim)
    output = (
        output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
    )  # (batch, seq_len, dim)

    return output, unnorm_attn_weight


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout=0.0, **kwargs):
        super(MultiheadAttention, self).__init__()
        assert dim % n_head == 0, (
            "Dimensionality must be divisible by the number of heads"
        )
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.num_heads = n_head
        self.head_dim = dim // n_head
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        # Linear projections for query, key, value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Call the multi_head_attention_forward function
        output, attn_weights = multi_head_attention_forward(
            query, key, value, self.num_heads, attn_mask
        )

        # Final output projection

        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_first = norm_first

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            out_src, attn_weight = self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask
            )
            src = src + out_src
            src = src + self._ff_block(self.norm2(src))
        else:
            out_src, attn_weight = self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask
            )
            src = self.norm1(src + out_src)
            src = self.norm2(src + self._ff_block(src))
        return src, attn_weight

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, attn_weihgts = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(x), attn_weihgts

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.encoder = nn.ModuleList([])
        self.encoder_layer = TransformerEncoderLayer(
            cfg.in_ch,
            cfg.n_head,
            cfg.dim_forward,
            cfg.dropout,
            F.gelu,
        )

        for i in range(num_layers):
            self.encoder.append(self.encoder_layer)

    def forward(self, src):
        attn_weihgts = []
        for i, layer in enumerate(self.encoder):
            src, attn_weihgt = layer(src)
            attn_weihgts.append(attn_weihgt.cpu().detach().numpy())
        return src, attn_weihgts


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=cfg.in_ch,
            nhead=cfg.n_head,
            dim_feedforward=cfg.dim_forward,
            dropout=cfg.dropout,
            activation=F.gelu,
        )
        self.transformer_encoder = TransformerEncoder(num_layers=1)
        self.attn_weights = None

    def forward(self, x):
        output, attn_weights = self.transformer_encoder(x)
        self.attn_weights = attn_weights
        return output
