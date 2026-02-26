# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.flow import FlowPrior


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        z: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        x = self.mask * z + (1 - self.mask) * (z * torch.exp(self.scale_net(self.mask * z) + self.translation_net(self.mask * z)))
        log_det_J = torch.sum((1-self.mask) * self.scale_net(self.mask * z), dim=1)
        return x, log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        z = self.mask * x + (1 - self.mask) * ((x - self.translation_net(self.mask * x)) * torch.exp(-self.scale_net(self.mask * x)))
        log_det_J = -torch.sum((1 - self.mask)*self.scale_net(self.mask * x), dim=1)
        return z, log_det_J

class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        z: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))
    
class FlowPrior(nn.Module):
    def __init__(self, M, num_flows):
        """
        Define a flow based prior distribution
        
        Parameters:
        M: [int]
            Dimension of the latent space.
        
        num_flows: [int]
            Amount of transformations to apply to the base distribution
        """
        super().__init__()
        
        self.base = GaussianBase(M)
        # Two complementary masks                                                             
        mask1 = torch.zeros(M)
        mask1[::2] = 1          # [1, 0, 1, 0, ...]

        mask2 = torch.ones(M)
        mask2[::2] = 0           # [0, 1, 0, 1, ...]
        
        hidden_dim_size = 128
        transformations = []
        for i in range(num_flows):
            mask = mask1 if i % 2 == 0 else mask2
            s_net = nn.Sequential(nn.Linear(M, hidden_dim_size), nn.ReLU(), nn.Linear(hidden_dim_size, M))
            t_net = nn.Sequential(nn.Linear(M, hidden_dim_size), nn.ReLU(), nn.Linear(hidden_dim_size, M))
            transformations.append(MaskedCouplingLayer(s_net, t_net, mask))
        
        self.flow = Flow(self.base, transformations)
        
    # Flow layers
    def forward(self):
        return self.flow


class MoGPrior(nn.Module):
    """Mixture of Gaussians prior (Exercise 1.6)."""
    def __init__(self, M, num_components):
        super().__init__()
        self.means = nn.Parameter(torch.randn(num_components, M))
        self.log_stds = nn.Parameter(torch.zeros(num_components, M))
        self.logits = nn.Parameter(torch.zeros(num_components))

    def forward(self):
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)



class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    """Gaussian output distribution for continuous MNIST (Exercise 1.7).

    fixed_std: if not None, use this fixed std for all pixels (not learned).
               if None, learn a per-pixel std.
    """
    def __init__(self, decoder_net, fixed_std=None):
        super().__init__()
        self.decoder_net = decoder_net
        self.log_std = nn.Parameter(torch.zeros(28, 28), requires_grad=(fixed_std is None))
        if fixed_std is not None:
            self.log_std.data.fill_(math.log(fixed_std))

    def forward(self, z):
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(mean, torch.exp(self.log_std)), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        prior = self.prior()
        # Use analytic KL when available (Gaussian prior), fall back to MC for MoG
        try:
            kl = td.kl_divergence(q, prior)
        except NotImplementedError:
            kl = q.log_prob(z) - prior.log_prob(z)
        return torch.mean(self.decoder(z).log_prob(x) - kl, dim=0)

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def evaluate(model, data_loader, device):
    """Compute mean ELBO on a dataset (Exercise 1.5)."""
    model.eval()
    total_elbo, n = 0.0, 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            total_elbo += model.elbo(x).item() * x.size(0)
            n += x.size(0)
    return total_elbo / n


def plot_posterior(model, data_loader, device, output_file):
    """Plot aggregate posterior samples coloured by class label (Exercise 1.5).

    For M > 2, projects onto first two PCA components.
    """

    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            zs.append(model.encoder(x).sample().cpu())
            labels.append(y)
    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    if zs.shape[1] > 2:
        zs = PCA(n_components=2).fit_transform(zs)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
    plt.colorbar(sc, ticks=range(10), label='Class')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Posterior plot saved to {output_file}')


def plot_prior_posterior(model, data_loader, device, output_file):
    """Plot prior samples against aggregate posterior samples.

    For M > 2, project to 2D with PCA.
    """
    model.eval()
    posterior = []
    n_total = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model.encoder(x).sample()
            posterior.append(z.cpu())
            n_total += z.size(0)
            # breakpoint()
        # breakpoint()
    posterior = torch.cat(posterior, dim=0).numpy()

    with torch.no_grad():
        prior = model.prior().sample(torch.Size([n_total])).cpu().numpy()

    if posterior.shape[1] > 2:
        both = np.concatenate([posterior, prior], axis=0)
        both_2d = PCA(n_components=2).fit_transform(both)
        n_post = len(posterior)
        posterior = both_2d[:n_post]
        prior = both_2d[n_post:]

    plt.figure(figsize=(8, 6))
    plt.scatter(prior[:, 0], prior[:, 1], s=1, alpha=0.35, label='Prior samples')
    plt.scatter(posterior[:, 0], posterior[:, 1], s=1, alpha=0.35, label='Posterior samples')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend(loc='best')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Prior vs posterior plot saved to {output_file}')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train',
                        choices=['train', 'sample', 'sample_mean', 'evaluate', 'plot_posterior', 'plot_prior_posterior'],
                        help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt',
                        help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png',
                        help='file to save samples in (default: %(default)s)')
    parser.add_argument('--posterior-plot', type=str, default='posterior.png',
                        help='file to save posterior plot in (default: %(default)s)')
    parser.add_argument('--prior-posterior-plot', type=str, default='prior_posterior.png',
                        help='file to save prior vs posterior plot in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N',
                        help='dimension of latent variable (default: %(default)s)')
    # Exercise 1.6: MoG prior / Exercise 2.6: Flow prior
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'],
                        help='prior distribution (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, metavar='N',
                        help='number of MoG components (default: %(default)s)')
    parser.add_argument('--flow-layers', type=int, default=8, metavar='N',
                        help='number of coupling layers in flow prior (default: %(default)s)')
    parser.add_argument('--flow-hidden', type=int, default=64, metavar='N',
                        help='hidden units in flow prior networks (default: %(default)s)')
    # Exercise 1.7: Gaussian decoder / continuous MNIST
    parser.add_argument('--decoder', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'],
                        help='decoder distribution (default: %(default)s)')
    parser.add_argument('--fixed-var', action='store_true',
                        help='use fixed variance (0.5) in Gaussian decoder instead of learned')
    parser.add_argument('--continuous', action='store_true',
                        help='use continuous MNIST instead of binarised')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST
    if args.continuous:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze()),
        ])
    else:
        threshold = 0.5
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (threshold < x).float().squeeze()),
        ])

    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'mog':
        prior = MoGPrior(M, args.num_components)
    elif args.prior == 'flow':
        prior = FlowPrior(M, n_transformations=args.flow_layers, n_hidden=args.flow_hidden)
    else:
        prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.decoder == 'gaussian':
        decoder = GaussianDecoder(decoder_net, fixed_std=0.5 if args.fixed_var else None)
    else:
        decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)

    else:
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        if args.mode == 'sample':
            with torch.no_grad():
                samples = model.sample(64).cpu()
                save_image(samples.view(64, 1, 28, 28), args.samples)

        elif args.mode == 'sample_mean':
            # Show mean of decoder instead of sample for Gaussian decoder (Ex 1.7)
            with torch.no_grad():
                z = model.prior().sample(torch.Size([64]))
                mean = model.decoder(z).mean.cpu()
                save_image(mean.view(64, 1, 28, 28), args.samples)

        elif args.mode == 'evaluate':
            elbo = evaluate(model, mnist_test_loader, device)
            print(f'Test ELBO: {elbo:.4f}')

        elif args.mode == 'plot_posterior':
            plot_posterior(model, mnist_test_loader, device, args.posterior_plot)

        elif args.mode == 'plot_prior_posterior':
            plot_prior_posterior(model, mnist_test_loader, device, args.prior_posterior_plot)
