import pytorch_lightning as tl
from pytorch_lightning.core.decorators import auto_move_data
import torch


class CNP(tl.LightningModule):
    def __init__(self, encoder, decoder, train_loader):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader

    @auto_move_data
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


class NP(tl.LightningModule):
    def __init__(self, encoder, decoder, train_loader, fixed_sigma=None, scale_kl=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.fixed_sigma = fixed_sigma
        self.scale_kl = scale_kl

    @auto_move_data
    def forward(self, context_x, context_y, target_x, target_y=None, use_mean_latent=True):
        # At train time, we infer with both context and targets
        # (and here target_x/y contains both the context and the targets)
        if target_y is not None:
            posterior = self.encoder(target_x, target_y)
            z = posterior.rsample() if not use_mean_latent else posterior.loc
        else:
            prior = self.encoder(context_x, context_y)
            z = prior.rsample() if not use_mean_latent else prior.loc
        return self.decoder(target_x, z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Could also add learning rate reduction w epoch number via torch.optim.lr_scheduler.StepLR
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def training_step(self, batch, batch_idx):
        context_x, context_y, target_x, target_y = batch
        out = self.forward(
            context_x, context_y, target_x, target_y=target_y, use_mean_latent=False
        )

        if self.fixed_sigma is None:
            mu, sigma = out # each (batch_size, n_targets, 1)
        else:
            # decoder can output either only mu (here) or mu and sigma (above) depending on output_sizes kwarg
            mu, sigma = out, torch.full_like(out, self.fixed_sigma) # each (batch_size, n_targets, 1)

        dist = torch.distributions.Normal(mu, sigma) 

        prior = self.encoder.forward(context_x, context_y)
        posterior = self.encoder.forward(target_x, target_y)

        batch_size, len_seq, _ = target_x.size()
        kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1)

        ll = dist.log_prob(target_y.squeeze(-1)).sum(dim=-1)
        elbo = ll - kl * len_seq if self.scale_kl else ll - kl
        nelbo = - elbo.mean()

        metrics = {'nelbo': nelbo, 'kl': kl.mean(), 'nll': -ll.mean(), 'batch_size': batch_size, 'sigma_output': sigma.mean()}
        output_dict = {"loss": nelbo, "log": metrics} # the "log" version of the metrics dict will be passed to the logger (e.g. tensorboard);
        output_dict.update(metrics) # also add all metrics to output dict, which will mean they are accessible as callback_metrics 
        return output_dict