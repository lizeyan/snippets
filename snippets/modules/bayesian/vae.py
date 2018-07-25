import torch.nn as nn
import torch.distributions as dist
import torch
import typing


def _sum_over_last_axis(x: torch.Tensor, times: int) -> torch.Tensor:
    y = x
    for i in range(times):
        y = x.sum(-1)
    return y


class VariationalAutoencoder(nn.Module):
    def __init__(self, *, variational_net: nn.Module,
                 generative_net: nn.Module,
                 x_layers: nn.ModuleDict,
                 x_dist_cls: typing.ClassVar[dist.Distribution],
                 z_layers: nn.ModuleDict,
                 z_dist_cls: typing.ClassVar[dist.Distribution],
                 z_event_ndims: int=1,
                 x_event_ndims: int=1, ):
        super().__init__()
        self.x_event_ndims = x_event_ndims
        self.z_event_ndims = z_event_ndims
        self.z_dist_cls = z_dist_cls
        self.z_layers = z_layers
        self.x_dist_cls = x_dist_cls
        self.variational_net = variational_net
        self.generative_net = generative_net
        self.x_layers = x_layers

    def elbo_sgvb(self, x, z_prior: dist.Distribution, y=None):
        z_posterior_dist = self.z_posterior(x, y)
        reparameterized_z_sample = z_posterior_dist.rsample()
        x_posterior_dist = self.x_posterior(reparameterized_z_sample, y)
        log_p_x_given_z = _sum_over_last_axis(x_posterior_dist.log_prob(x), self.x_event_ndims)
        log_p_z = _sum_over_last_axis(z_prior.log_prob(reparameterized_z_sample), self.z_event_ndims)
        log_p_z_given_x = _sum_over_last_axis(z_posterior_dist.log_prob(reparameterized_z_sample), self.z_event_ndims)
        return (log_p_x_given_z + log_p_z - log_p_z_given_x).mean()

    def z_posterior(self, x: torch.Tensor, y: torch.Tensor=None) -> dist.Distribution:
        hidden = self.variational_net((x, y)) if y is not None else self.variational_net(x)
        statistics = {}
        for key, item in self.z_layers.items():
            statistics[key] = item(hidden)
        return self.z_dist_cls(**statistics)

    def x_posterior(self, z: torch.Tensor, y: torch.Tensor=None) -> dist.Distribution:
        hidden = self.generative_net((z, y)) if y is not None else self.generative_net(z)
        statistics = {}
        for key, item in self.x_layers.items():
            statistics[key] = item(hidden)
        return self.x_dist_cls(**statistics)

    def forward(self, x, z_prior, y=None):
        return self.elbo_sgvb(x, z_prior, y)

    def encode(self, x, y=None):
        return self.z_posterior(x, y)

    def decode(self, z, y=None):
        return self.x_posterior(z, y)

    def reconstruct(self, x, y=None, sample_times: int =1) -> dist.Distribution:
        z_posterior_dist = self.z_posterior(x, y)
        z_samples = z_posterior_dist.sample((sample_times,))
        if y is not None:
            origin_size = y.size()
            y = y.unsqueeze_(0).expand(sample_times, *origin_size)
        x_posterior_dist = self.x_posterior(z_samples, y) if y is not None else None
        return x_posterior_dist


VAE = VariationalAutoencoder
