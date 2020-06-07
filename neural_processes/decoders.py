from typing import List

import torch
import torch.nn.functional as F

from neural_processes import base


class Decoder(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int]):
        super().__init__()
        self.mlp = base.MLP(input_size, output_sizes)

    def forward(self, x, r):
        seq_len = x.size()[1]

        # Give each target point a copy of the representation r to decode with
        inp = torch.cat([x, r.unsqueeze(dim=1).repeat([1, seq_len, 1])], dim=-1)

        out = self.mlp.forward(inp)

        mu, sigma = out[:, :, 0], out[:, :, 1]

        # Make the sigma positive and bounded below at 0.1
        sigma = 0.1 + 0.9 * F.softplus(sigma)

        return mu, sigma
