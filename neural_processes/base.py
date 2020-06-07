from typing import List

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int]):
        super().__init__()
        self.initial = torch.nn.Linear(input_size, output_sizes[0])
        num_hidden = len(output_sizes) - 2
        if num_hidden < 0:
            raise ValueError(f"output_sizes:{output_sizes} should have 2 or more elements")
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(output_sizes[i], output_sizes[i + 1]) for i in range(num_hidden)]
        )
        self.final = torch.nn.Linear(output_sizes[-2], output_sizes[-1])

    def forward(self, X):
        # Layers act on the final dimension, so share weights for points across the series
        mid = F.relu(self.initial(X))
        for layer in self.hidden:
            mid = F.relu(layer(mid))
        return self.final(mid)
