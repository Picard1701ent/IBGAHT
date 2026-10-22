import logging
from types import SimpleNamespace

import config
import torch
from torch import nn

from models.IBAHG import IBAHGConvBlock
from models.IBLoss import VariationalInfoBottleneck
from models.TransformerEncoder import TransformerBlock
from models.utils import Mlp

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


class IBAHGTBlock(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.transformer_block = TransformerBlock()
        self.IBAHGConvBlock = IBAHGConvBlock(weight)
        self.fusion_weight = nn.Parameter(torch.Tensor(cfg.node_num, 1))
        self.fusion_classification = Mlp(
            cfg.node_num * cfg.out_ch, cfg.hidden_ch, 2, cfg.dropout
        )
        self.classi_cri = nn.CrossEntropyLoss(weight=weight)
        self.miloss = VariationalInfoBottleneck()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.fusion_weight, 0.5)

    def forward(self, x, H, labels):
        IBAHGConvBlock_out, IBAHGConvBlock_loss = self.IBAHGConvBlock(x, H, labels)
        transformer_out = self.transformer_block(x)
        out = transformer_out * self.fusion_weight + IBAHGConvBlock_out
        fusion_classi = self.fusion_classification(out.flatten(-2, -1))
        fusion_classi = torch.softmax(fusion_classi, dim=-1)
        fusion_loss = self.classi_cri(fusion_classi, labels)
        out, _, mi_loss = self.miloss(out)
        loss = IBAHGConvBlock_loss + fusion_loss + cfg.beta2 * mi_loss

        return out, loss


class IBAHGT(nn.Module):
    def __init__(self, num_layer=2, weight=None):
        super().__init__()
        self.classifier = Mlp(cfg.node_num * cfg.out_ch, cfg.hidden_ch, 2, cfg.dropout)
        self.backbone = nn.ModuleList([])
        self.weight = weight
        for i in range(num_layer):
            self.backbone.append(IBAHGTBlock(weight=self.weight))
        self.attn_weights = []
        self.fusion_weights = []

    def forward(self, x, H, labels):
        self.attn_weights = []
        self.fusion_weights = []

        src = x.clone()
        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        for block in self.backbone:
            x, loss = block(x, H, labels)
            total_loss = total_loss + loss
            x = x + src
            self.attn_weights.append(block.transformer_block.attn_weights)
            self.fusion_weights.append(block.fusion_weight.data)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        out = nn.functional.softmax(x, dim=-1)
        out = self.classifier(x)
        return out, loss
