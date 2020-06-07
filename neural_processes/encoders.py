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
