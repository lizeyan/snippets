import unittest

import torch
import torch.distributions as dist
import torch.nn as nn

from snippets.modules import Lambda
from snippets.modules.bayesian import VariationalAutoEncoder


class TestVAE(unittest.TestCase):
    def test_vae(self):
        x_dims = 784
        z_dims = 2
        hidden_dims = 128
        eps = 1e-4
        vae = VariationalAutoEncoder(
            encoder=nn.Sequential(
                nn.Linear(x_dims, hidden_dims), nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            ),
            decoder=nn.Sequential(
                nn.Linear(z_dims, hidden_dims), nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            ),
            z_dist_class=lambda *args, **kwargs: dist.Independent(dist.Normal(*args, **kwargs), 1),
            x_dist_class=lambda *args, **kwargs: dist.Independent(dist.Normal(*args, **kwargs), 1),
            z_dist_params=nn.ModuleDict({
                'loc': nn.Linear(hidden_dims, z_dims),
                'scale': nn.Sequential(
                    nn.Linear(hidden_dims, z_dims),
                    nn.Softplus(),
                    Lambda(lambda _: _ + eps)
                )
            }),
            x_dist_params=nn.ModuleDict({
                'loc': nn.Linear(hidden_dims, x_dims),
                'scale': nn.Sequential(
                    nn.Linear(hidden_dims, x_dims),
                    nn.Softplus(),
                    Lambda(lambda _: _ + eps)
                )
            }),
            z_prior_dist_class=lambda *args, **kwargs: dist.Independent(dist.Normal(*args, **kwargs), 1),
            z_prior_dist_params=nn.ParameterDict({
                'loc': nn.Parameter(torch.zeros(z_dims, dtype=torch.float32)),
                'scale': nn.Parameter(torch.ones(z_dims, dtype=torch.float32))
            }),
            beta=10,
        )

        vae.train()

        x = torch.randn(11, 5, x_dims)
        q_z = vae.encode(x, n_group_dims=2)  # dist.Distribution
        self.assertEqual(q_z.event_shape, (z_dims,))
        self.assertEqual(q_z.batch_shape, (11, 5,))
        z_samples = q_z.rsample(sample_shape=(7, 3))
        self.assertEqual(z_samples.size(), (7, 3, 11, 5, z_dims))
        p_x = vae.decode(z_samples, n_group_dims=4)
        self.assertEqual(p_x.event_shape, (x_dims,))
        self.assertEqual(p_x.batch_shape, (7, 3, 11, 5,))
        ll = vae.log_likelihood(x, n_group_dims=2)
        elbo = vae.evidence_lower_bound(x, n_group_dims=2)
        rp = vae.reconstruction_probability(x, n_group_dims=2)
        self.assertEqual(ll.size(), (11, 5))
        self.assertEqual(elbo.size(), (11, 5))
        self.assertEqual(rp.size(), (11, 5))

        self.assertEqual(vae.generative().batch_shape, ())
        self.assertEqual(vae.generative().event_shape, (x_dims,))
        x_dist, z_dist, z_samples = vae.reconstruct(x, n_group_dims=2)
        self.assertEqual(x_dist.event_shape, (x_dims,))
        self.assertEqual(z_dist.event_shape, (z_dims,))
        self.assertEqual(x_dist.batch_shape, (11, 5))
        self.assertEqual(x_dist.batch_shape, (11, 5))
        self.assertEqual(z_samples.size(), (11, 5, z_dims))
        x_dist, z_dist, z_samples = vae(x, n_group_dims=2)
        self.assertEqual(x_dist.event_shape, (x_dims,))
        self.assertEqual(z_dist.event_shape, (z_dims,))
        self.assertEqual(x_dist.batch_shape, (11, 5))
        self.assertEqual(x_dist.batch_shape, (11, 5))
        self.assertEqual(z_samples.size(), (11, 5, z_dims))

        self.assertIsNotNone(z_samples.grad_fn)

        vae.eval()
        x_dist, z_dist, z_samples = vae(x, n_group_dims=2)
        self.assertIsNone(z_samples.grad_fn)

