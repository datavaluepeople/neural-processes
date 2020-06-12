from typing import List

import pytorch_lightning as tl
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


class Encoder(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int]):
        super().__init__()
        self.mlp = MLP(input_size, output_sizes)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=-1)

        r = self.mlp.forward(inp)

        # Collapse r to a single output per input series
        return r.mean(dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int]):
        super().__init__()
        self.mlp = MLP(input_size, output_sizes)

    def forward(self, x, r):
        seq_len = x.size()[1]

        # Give each target point a copy of the representation r to decode with
        inp = torch.cat([x, r.unsqueeze(dim=1).repeat([1, seq_len, 1])], dim=-1)

        out = self.mlp.forward(inp)

        mu, sigma = out[:, :, 0], out[:, :, 1]

        # Make the sigma positive and bounded below at 0.1
        sigma = 0.1 + 0.9 * F.softplus(sigma)

        return mu, sigma


class Model(tl.LightningModule):
    def __init__(self, encoder, decoder, train_loader):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader

    def forward(self, context_x, context_y, target_x):
        r = self.encoder(context_x, context_y)
        mu, sigma = self.decoder(target_x, r)
        return mu, sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Could also add learning rate reduction w epoch number via torch.optim.lr_scheduler.StepLR
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def training_step(self, batch, batch_idx):
        context_x, context_y, target_x, target_y = batch
        mu, sigma = self.forward(context_x, context_y, target_x)
        dist = torch.distributions.normal.Normal(mu, sigma)
        loss = - dist.log_prob(target_y.squeeze(-1))
        return {"loss": loss.mean()}
