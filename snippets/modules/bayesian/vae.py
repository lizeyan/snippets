import math
from typing import Union, Type, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

VAE_RETURN_TYPE = Tuple[dist.Distribution, dist.Distribution, torch.Tensor]


class VariationalAutoEncoder(nn.Module):
    def __init__(
            self,
            encoder: nn.Module, z_dist_class: Type[dist.Distribution], z_dist_params: nn.ModuleDict,
            decoder: nn.Module, x_dist_class: Type[dist.Distribution], x_dist_params: nn.ModuleDict,
            z_prior_dist_class: Type[dist.Distribution], z_prior_dist_params: nn.ParameterDict,
            beta: float = 1.,
    ):
        """
        :param encoder: f(x)
        :param decoder: g(z)
        :param beta: for beta-VAE, ELBO = ReconstructionLoss - beta * KL[q(z|x)||p(z)]
        """
        super(VariationalAutoEncoder, self).__init__()
        self.beta = beta
        self._encoder = encoder
        self._decoder = decoder
        self._z_dist_class = z_dist_class
        self._z_dist_params = z_dist_params  # type: nn.ModuleDict
        self._x_dist_class = x_dist_class  # type: nn.ModuleDict
        self._x_dist_params = x_dist_params
        self.z_prior_dist_class = z_prior_dist_class
        self.z_prior_dist_params = z_prior_dist_params

        self.variational = self.encode

    def _sample(self, distribution: dist.Distribution, sample_shape: Union[torch.Size, tuple] = torch.Size()):
        if self.training:
            return distribution.rsample(sample_shape=sample_shape)
        else:
            return distribution.sample(sample_shape=sample_shape)

    @property
    def z_prior_dist(self) -> dist.Distribution:
        return self.z_prior_dist_class(**self.z_prior_dist_params)

    def forward(self, x, sample_shape=(), n_group_dims=0) -> VAE_RETURN_TYPE:
        return self.reconstruct(x, sample_shape, n_group_dims)

    def reconstruct(self, x_samples, sample_shape=(), n_group_dims=0) -> VAE_RETURN_TYPE:
        z_dist = self.encode(x_samples, n_group_dims=n_group_dims)
        z_samples = self._sample(z_dist, sample_shape)
        x_dist = self.decode(z_samples, n_group_dims=len(sample_shape) + n_group_dims)
        return x_dist, z_dist, z_samples

    def generative(self, sample_shape=()):
        z_samples = self._sample(self.z_prior_dist, sample_shape)
        x_dist = self.decode(z_samples=z_samples, n_group_dims=len(sample_shape))
        return x_dist

    def encode(self, x_samples, n_group_dims=0) -> dist.Distribution:
        """
        :param n_group_dims:
        :param x_samples: sample_shape + (batch_size, ) + event_shape
        :return:
        """
        group_shape = x_samples.size()[:n_group_dims]
        x = x_samples.view((-1,) + x_samples.size()[n_group_dims:])
        shared = self._encoder(x)
        shared = shared.view(group_shape + shared.size()[1:])
        z_params = {param_name: param_module(shared) for param_name, param_module in self._z_dist_params.items()}
        return self._z_dist_class(**z_params)

    def decode(self, z_samples, n_group_dims=0) -> dist.Distribution:
        """
        :param n_group_dims:
        :param z_samples: sample_shape + (batch_size, ) + event_shape
        :return:
        """
        group_shape = z_samples.size()[:n_group_dims]
        z = z_samples.view((-1,) + z_samples.size()[n_group_dims:])
        shared = self._decoder(z)
        shared = shared.view(group_shape + shared.size()[1:])
        x_params = {param_name: param_module(shared) for param_name, param_module in self._x_dist_params.items()}
        return self._x_dist_class(**x_params)

    def log_likelihood(self, x_samples, n_samples=1, n_group_dims=1):
        """
        emulate LL by importance sampling
        """
        z_dist = self.encode(x_samples, n_group_dims=n_group_dims)
        z_samples = self._sample(z_dist, (n_samples,))
        x_dist = self.decode(z_samples, n_group_dims=1 + n_group_dims)
        p_x_z = x_dist.log_prob(x_samples)
        p_z = self.z_prior_dist.log_prob(z_samples)
        q_z_x = z_dist.log_prob(z_samples)
        log_likelihood = p_x_z + p_z - q_z_x
        return torch.logsumexp(log_likelihood, dim=0) - math.log(log_likelihood.shape[0])

    def evidence_lower_bound(self, x_samples, n_samples=1, n_group_dims=0):
        q_z_given_x = self.variational(x_samples, n_group_dims=n_group_dims)
        z_samples = self._sample(q_z_given_x, (n_samples,))
        p_x_given_z = self.decode(z_samples, n_group_dims=1 + n_group_dims)
        log_likelihood = p_x_given_z.log_prob(x_samples) + self.z_prior_dist.log_prob(z_samples)
        entropy_item = q_z_given_x.log_prob(z_samples)
        elbo = log_likelihood - entropy_item
        return torch.mean(elbo, dim=0)

    def reconstruction_probability(self, x_samples, n_samples=1, n_group_dims=0):
        q_z_given_x = self.variational(x_samples, n_group_dims=n_group_dims)
        z_samples = self._sample(q_z_given_x, (n_samples,))
        p_x_given_z = self.decode(z_samples, n_group_dims=1 + n_group_dims)
        log_likelihood = p_x_given_z.log_prob(x_samples)
        return torch.mean(log_likelihood, dim=0)


VAE = VariationalAutoEncoder

__all__ = [
    'VAE', 'VariationalAutoEncoder'
]
