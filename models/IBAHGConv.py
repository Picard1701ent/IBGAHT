import logging
from types import SimpleNamespace

import config
import torch
from torch import einsum, nn
from torch.nn import Parameter

from models.loss_module import VariationalInfoBottleneck
from models.utils import Mlp

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


class IBAHGConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, bias=True) -> None:
        super().__init__()
        self.drop_out = dropout
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter("bias", None)

        # self.bn = nn.BatchNorm1d(out_ch)
        self.bn = nn.LayerNorm(out_ch)
        self.gelu = nn.GELU()  # nn.ReLU(inplace=True)
        # self.H_attn = SelfAttention(in_ch,in_ch//4)
        self.node_weight_learner = nn.Linear(cfg.in_ch, 1)
        self.reset_parameters()
        self.info_n2e = VariationalInfoBottleneck()
        self.info_e2n = VariationalInfoBottleneck()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H):
        assert len(x.shape) == 3, "the input of HyperConv should be N * V * C"
        # H = H * gaussian_distance
        y = einsum("nvc,co->nvo", x, self.theta)  # y: [batch, node, target_features]
        node_weight = (
            self.node_weight_learner(x.flatten(0, 1))
            .sigmoid()
            .reshape(x.shape[0], x.shape[1], -1)
        )  # B*N*1
        if len(node_weight.shape) == 2:
            node_weight = node_weight.unsqueeze(0)
        Dv = torch.diag_embed(
            1.0 / H.sum(1), dim1=-2, dim2=-1
        )  # H: [batch, node, edge] Dv: [batch, node, node]
        HDv = einsum("nkv,nve->nke", Dv, H)  # HDv = [batch, node, edge]
        He = H * node_weight  # B, N, E * B * N * 1
        De = torch.diag_embed(
            1.0 / He.sum(2), dim1=-2, dim2=-1
        )  # De: [batch, edge, edge]
        HDe = einsum("nve,nek->nvk", H, De)  # HDe: [batch, node, edge]
        e_fts = einsum(
            "nvc,nve->nec", y, HDe
        )  # y: [batch, node, target_features] HDe: [batch, node, edge] -> e_fts: [batch, edge, target_features]
        e_fts, _, e_loss = self.info_e2n(e_fts)
        y = einsum("nec,nve->nvc", e_fts, HDv)  # y: [batch, node, target_features]
        y = y + self.bias.unsqueeze(0).unsqueeze(0)

        # y = self.pooling(y)
        y, _, n_loss = self.info_n2e(y)
        return self.gelu(y), e_fts, e_loss, n_loss


class IBAHGConvBlock(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.eclassifier = Mlp(
            cfg.mlp_in_features, cfg.mlp_hidden_features, cfg.mlp_out_features
        )
        self.IBAHGConv = IBAHGConv(cfg.in_ch, cfg.out_ch, cfg.dropout)

        self.nclassifier = Mlp(
            cfg.mlp_in_features, cfg.mlp_hidden_features, cfg.mlp_out_features
        )
        self.classi_cri = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x, H, labels):
        out, e_out, e_miloss, n_miloss = self.IBAHGConv(x, H)

        e_classi = self.eclassifier(torch.flatten(e_out, start_dim=-2, end_dim=-1))
        e_classi = torch.softmax(e_classi, dim=-1)
        e_classi_loss = self.classi_cri(e_classi, labels)
        n_classi = self.nclassifier(torch.flatten(out, start_dim=-2, end_dim=-1))
        n_classi = torch.softmax(n_classi, dim=-1)
        n_classi_loss = self.classi_cri(n_classi, labels)
        loss = (
            e_classi_loss + cfg.beta1 * e_miloss + cfg.beta1 * n_miloss + n_classi_loss
        )
        return out, loss
