from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

import config
from pytorch_net.net import Mixture_Gaussian_reparam

cfg = SimpleNamespace(**vars(config))

device = torch.device(cfg.device)


class EncoderNetwork(nn.Module):
    def __init__(self, input_dim=90 * 90, latent_dim=256, n_components=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.mean_scale = nn.Linear(512, 2 * latent_dim * n_components)
        self.weight_logits = nn.Linear(512, n_components)

        self.prior = Mixture_Gaussian_reparam(
            Z_size=latent_dim,
            n_components=n_components,
            is_reparam=False,
            reparam_mode="diag",
            is_cuda=torch.cuda.is_available(),
        )

    def reparameterize_mixture(self, x):
        x = self.encoder(x)

        mean_scale = self.mean_scale(x)
        weight_logits = self.weight_logits(x)

        batch_size = x.shape[0]
        mean_scale = mean_scale.view(batch_size, 2 * self.latent_dim, self.n_components)
        mean_list = mean_scale[:, : self.latent_dim]
        scale_list = F.softplus(mean_scale[:, self.latent_dim :], beta=1) + 0.01

        dist = Mixture_Gaussian_reparam(
            mean_list=mean_list,
            scale_list=scale_list,
            weight_logits=weight_logits,
            is_reparam=True,
            reparam_mode="diag",
            is_cuda=x.is_cuda,
        )

        return dist

    def forward(self, x):
        dist = self.reparameterize_mixture(x)

        Z_l = dist.rsample()

        Z_logit = dist.log_prob(Z_l).sum(-1)

        prior_logit = self.prior.log_prob(Z_l).sum(-1)

        ixz = (Z_logit - prior_logit).mean()

        return Z_l, ixz


class DecoderNetwork(nn.Module):
    def __init__(self, latent_dim=256, output_dim=90 * 90):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 90, 90)


class VariationalInfoBottleneck(nn.Module):
    def __init__(self, input_dim=90 * 90, latent_dim=256, n_components=5, beta=0.01):
        super().__init__()
        self.encoder = EncoderNetwork(input_dim, latent_dim, n_components)
        self.decoder = DecoderNetwork(latent_dim, input_dim)

    def forward(self, x):
        Z_l, ixz = self.encoder(x)
        x_recon = self.decoder(Z_l)
        return x_recon, None, ixz + 0.001 * F.mse_loss(x_recon, x)
