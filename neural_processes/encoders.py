from typing import List

import torch

from neural_processes import base


class Deterministic(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int]):
        super().__init__()
        self.mlp = base.MLP(input_size, output_sizes)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=-1)

        r = self.mlp.forward(inp)

        # Collapse r to a single output per input series
        return r.mean(dim=1)


class Latent(torch.nn.Module):
    def __init__(self, encoder_net, latent_params_net):
        super().__init__()
        self.encoder = encoder_net
        self.latent_params = latent_params_net

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=-1)

        r = self.encoder.forward(inp)

        # Collapse r to a single output per input series
        r = r.mean(dim=1)

        latent_params = self.latent_params(r)
        latent_params = latent_params.view((latent_params.size()[0], 2, -1))
        mu, log_sigma = latent_params[:, 0], latent_params[:, 1]

        # Make the sigma positive and bounded below at 0.1
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return torch.distributions.Normal(mu, sigma)
