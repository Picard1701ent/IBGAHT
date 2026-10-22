from types import SimpleNamespace

from torch import nn

import config

cfg = SimpleNamespace(**vars(config))


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=2, drop_rate=0.1
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = 2 * in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, X):
        return self.net(X)
