from snippets.modules import VAE, MLP, Lambda
import torch
import torch.nn as nn
import torch.distributions as dist
import unittest


class TestVAE(unittest.TestCase):
    def test_vae(self):
        model = VAE(
            variational_net=MLP(784, [256, 100]),
            generative_net=MLP(2, [100, 256]),
            z_layers=nn.ModuleDict({
                "loc": nn.Sequential(nn.Linear(100, 2)),
                "scale": nn.Sequential(nn.Linear(100, 2), nn.Softplus(), )
            }),
            x_layers=nn.ModuleDict({
                "probs": nn.Sequential(nn.Linear(256, 784), nn.Sigmoid())
            }),
            x_dist_cls=dist.Bernoulli,
            z_dist_cls=dist.Normal,
        )
        z_prior = dist.Normal(torch.Tensor((0.,)), torch.Tensor((1.,)))
        x = torch.Tensor(32, 784)
        z = model.encode(x)
        x = model.decode(z.sample())
        model(x.mean, z_prior)
        reconstructed = model.reconstruct(x.mean, sample_times=12)
        self.assertEqual(reconstructed.mean.size(), torch.Size((12, 32, 784)))

    def test_cvae(self):
        model = VAE(
            variational_net=nn.Sequential(Lambda(lambda x: torch.cat([x[0], x[1]], -1)), MLP(784 + 10, [256, 100])),
            generative_net=nn.Sequential(Lambda(lambda x: torch.cat([x[0], x[1]], -1)), MLP(2 + 10, [100, 256])),
            z_layers=nn.ModuleDict(
                {
                    "loc": nn.Sequential(nn.Linear(100, 2)),
                    "scale": nn.Sequential(nn.Linear(100, 2), nn.Softplus(), )
                }),
            x_layers=nn.ModuleDict({
                "probs": nn.Sequential(nn.Linear(256, 784), nn.Sigmoid())
            }),
            x_dist_cls=dist.Bernoulli,
            z_dist_cls=dist.Normal,
        )
        z_prior = dist.Normal(torch.Tensor((0.,)), torch.Tensor((1.,)))
        x = torch.Tensor(32, 784)
        y = torch.Tensor(32, 10)
        z = model.encode(x, y)
        x_dist = model.decode(z.sample(), y)
        model(x_dist.mean, z_prior, y)
        reconstructed = model.reconstruct(x_dist.mean, sample_times=12, y=y)
        self.assertEqual(reconstructed.mean.size(), torch.Size((12, 32, 784)))
