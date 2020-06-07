import pytorch_lightning as tl
import torch


class CNP(tl.LightningModule):
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
