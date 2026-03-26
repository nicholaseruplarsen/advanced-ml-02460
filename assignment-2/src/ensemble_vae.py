# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt

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


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


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

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

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
    
def curve_energy(curve, decoder): # descritizing the energy integral as a sum (lol). the metric is encoded in the decoder network, because it induces the geometry. so we dont need to bother with position on the manifold
    decoded = decoder(curve).mean
    diffs = decoded[1:] - decoded[:-1]
    return (diffs ** 2).sum()

def compute_geodesic(a, b, decoder, num_points=20, num_epochs=500, lr=1e-2):
    t = torch.linspace(0, 1, num_points, device=a.device).unsqueeze(1)
    curve = (1 - t) * a + t * b # torch magic to make the curve a well designed tensor for fast fast operations
    
    interior = curve[1:-1].clone().detach().requires_grad_(True) # have the optimizer look only at the interior points of the pw linear curve, not the fixed start and end points
    optimizer = torch.optim.Adam([interior], lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        full_curve = torch.cat([a.unsqueeze(0), interior, b.unsqueeze(0)], dim=0) # connect the start and end points back
        energy = curve_energy(full_curve, decoder)
        energy.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_curve = torch.cat([a.unsqueeze(0), interior, b.unsqueeze(0)], dim=0)
    return final_curve

def ensemble_curve_energy(curve,decoders:list,num_mc_samples:int = 16):
    energy = 0.0
    for _ in range(num_mc_samples):
        l,k=torch.randint(len(decoders),(1,)),torch.randint(len(decoders),(1,))
        decoded_l,decoded_k=decoders[l](curve[:-1]).mean,decoders[k](curve[1:]).mean
        energy+=(decoded_l-decoded_k).pow(2).sum()
    return energy/num_mc_samples

def compute_ensemble_geodesic(a, b, decoders, num_points=20, num_epochs= 500, lr = 1e-2):
    t = torch.linspace(0, 1, num_points, device=a.device).unsqueeze(1)
    curve = (1 - t) * a + t * b # torch magic to make the curve a well designed tensor for fast fast operations
    
    interior = curve[1:-1].clone().detach().requires_grad_(True) # have the optimizer look only at the interior points of the pw linear curve, not the fixed start and end points
    optimizer = torch.optim.Adam([interior], lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        full_curve = torch.cat([a.unsqueeze(0), interior, b.unsqueeze(0)], dim=0)  # connect the start and end points back
        energy = ensemble_curve_energy(full_curve, decoders)
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        final_curve = torch.cat([a.unsqueeze(0), interior, b.unsqueeze(0)], dim=0)
    return final_curve

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

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "ensemble", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    # Anchor all relative paths to the assignment-2 folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        os.path.join(base_dir, "data"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        os.path.join(base_dir, "data"),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":
        experiments_folder = os.path.join(base_dir, args.experiment_folder)
        os.makedirs(experiments_folder, exist_ok=True)

        # Phase 1: train encoder + first decoder jointly
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("Training encoder + decoder 0")
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)

        # Save encoder and first decoder
        torch.save(model.encoder.state_dict(), f"{experiments_folder}/encoder.pt")
        torch.save(model.decoder.state_dict(), f"{experiments_folder}/decoder_0.pt")

        # Phase 2: freeze encoder, train additional decoders
        for param in model.encoder.parameters():
            param.requires_grad = False

        for d in range(1, args.num_decoders):
            decoder = GaussianDecoder(new_decoder()).to(device)
            # Build a temporary VAE with the frozen encoder and new decoder
            tmp = VAE(GaussianPrior(M), decoder, model.encoder).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            print(f"Training decoder {d}")
            train(tmp, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
            torch.save(decoder.state_dict(), f"{experiments_folder}/decoder_{d}.pt")

    elif args.mode == "sample":
        experiments_folder = os.path.join(base_dir, args.experiment_folder)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.encoder.load_state_dict(torch.load(f"{experiments_folder}/encoder.pt"))
        model.decoder.load_state_dict(torch.load(f"{experiments_folder}/decoder_0.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), os.path.join(base_dir, args.samples))

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0),
                os.path.join(base_dir, "reconstruction_means.png"),
            )

    elif args.mode == "eval":
        experiments_folder = os.path.join(base_dir, args.experiment_folder)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.encoder.load_state_dict(torch.load(f"{experiments_folder}/encoder.pt"))
        model.decoder.load_state_dict(torch.load(f"{experiments_folder}/decoder_0.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        experiments_folder = os.path.join(base_dir, args.experiment_folder)

        # Load encoder
        encoder = GaussianEncoder(new_encoder()).to(device)
        encoder.load_state_dict(torch.load(f"{experiments_folder}/encoder.pt"))
        encoder.eval()

        # Load all available decoders
        all_decoders = []
        for d in range(args.num_decoders):
            path = f"{experiments_folder}/decoder_{d}.pt"
            if not os.path.exists(path):
                break
            dec = GaussianDecoder(new_decoder()).to(device)
            dec.load_state_dict(torch.load(path))
            dec.eval()
            all_decoders.append(dec)
        print(f"Loaded {len(all_decoders)} decoders")

        # Encode test data
        with torch.no_grad():
            all_z, all_labels = [], []
            for batch_x, batch_y in mnist_test_loader:
                z = encoder(batch_x.to(device)).mean
                all_z.append(z.cpu())
                all_labels.append(batch_y)
            all_z = torch.cat(all_z)
            all_labels = torch.cat(all_labels)

        # Fixed random pairs
        torch.manual_seed(0)
        pairs = torch.randperm(len(all_z))[:2 * args.num_curves].reshape(-1, 2)

        # Plot for K=1 (Part A) and K=all (Part B) side by side
        for K in [1, len(all_decoders)]:
            decoders = all_decoders[:K]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.3, s=10)

            for i in tqdm(range(len(pairs)), desc=f"Geodesics K={K}"):
                a = all_z[pairs[i, 0]].to(device)
                b = all_z[pairs[i, 1]].to(device)
                if K == 1:
                    curve = compute_geodesic(a, b, decoders[0], num_points=args.num_t)
                else:
                    curve = compute_ensemble_geodesic(a, b, decoders, num_points=args.num_t)
                curve_np = curve.detach().cpu().numpy()
                ax.plot(curve_np[:, 0], curve_np[:, 1], 'r-', linewidth=1.5)

            ax.set_title(f'Geodesics (K={K} decoders)')
            plt.savefig(f"{experiments_folder}/geodesics_K{K}.png", dpi=150)
            plt.close(fig)

    elif args.mode == "cov":
        import numpy as np

        num_reruns = args.num_reruns
        num_decoders = args.num_decoders
        num_curves = args.num_curves

        # Train num_reruns independent ensemble VAEs
        for run in range(num_reruns):
            run_folder = os.path.join(base_dir, args.experiment_folder, f"run_{run}")
            if os.path.exists(f"{run_folder}/decoder_0.pt"):
                print(f"Run {run} already trained, skipping")
                continue

            os.makedirs(run_folder, exist_ok=True)
            torch.manual_seed(run)

            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            print(f"Run {run}: training encoder + decoder 0")
            train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
            torch.save(model.encoder.state_dict(), f"{run_folder}/encoder.pt")
            torch.save(model.decoder.state_dict(), f"{run_folder}/decoder_0.pt")

            for param in model.encoder.parameters():
                param.requires_grad = False

            for d in range(1, num_decoders):
                decoder = GaussianDecoder(new_decoder()).to(device)
                tmp = VAE(GaussianPrior(M), decoder, model.encoder).to(device)
                optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
                print(f"Run {run}: training decoder {d}")
                train(tmp, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
                torch.save(decoder.state_dict(), f"{run_folder}/decoder_{d}.pt")

        # Fix test pairs (same across all runs)
        torch.manual_seed(999)
        test_xs = []
        for batch_x, _ in mnist_test_loader:
            test_xs.append(batch_x)
        test_xs = torch.cat(test_xs).to(device)
        pair_indices = torch.randperm(len(test_xs))[:2 * num_curves].reshape(-1, 2)

        # Compute distances
        eucl_dists = np.zeros((num_reruns, num_curves))
        geod_dists = {K: np.zeros((num_reruns, num_curves)) for K in range(1, num_decoders + 1)}

        for run in range(num_reruns):
            run_folder = os.path.join(base_dir, args.experiment_folder, f"run_{run}")

            encoder = GaussianEncoder(new_encoder()).to(device)
            encoder.load_state_dict(torch.load(f"{run_folder}/encoder.pt"))
            encoder.eval()

            decoders = []
            for d in range(num_decoders):
                dec = GaussianDecoder(new_decoder()).to(device)
                dec.load_state_dict(torch.load(f"{run_folder}/decoder_{d}.pt"))
                dec.eval()
                decoders.append(dec)

            with torch.no_grad():
                z_means = encoder(test_xs).mean

            for p in range(num_curves):
                i, j = pair_indices[p]
                zi, zj = z_means[i], z_means[j]
                eucl_dists[run, p] = (zi - zj).norm().item()

                for K in range(1, num_decoders + 1):
                    if K == 1:
                        curve = compute_geodesic(zi, zj, decoders[0], num_points=args.num_t)
                        with torch.no_grad():
                            d = curve_energy(curve, decoders[0]).sqrt().item()
                    else:
                        curve = compute_ensemble_geodesic(zi, zj, decoders[:K], num_points=args.num_t)
                        with torch.no_grad():
                            d = ensemble_curve_energy(curve, decoders[:K]).sqrt().item()
                    geod_dists[K][run, p] = d
                    print(f"Run {run}, pair {p}, K={K}: eucl={eucl_dists[run,p]:.4f} geod={d:.4f}")

        # CoV
        eucl_cov = eucl_dists.std(axis=0) / (eucl_dists.mean(axis=0) + 1e-8)

        fig, ax = plt.subplots(figsize=(6, 4))
        Ks = list(range(1, num_decoders + 1))
        geod_covs = []
        for K in Ks:
            g_cov = geod_dists[K].std(axis=0) / (geod_dists[K].mean(axis=0) + 1e-8)
            geod_covs.append(g_cov.mean())
        ax.plot(Ks, [eucl_cov.mean()] * len(Ks), 'o--', label='Euclidean')
        ax.plot(Ks, geod_covs, 's-', label='Geodesic')
        ax.set_xlabel('Ensemble decoders (K)')
        ax.set_ylabel('Mean CoV')
        ax.set_xticks(Ks)
        ax.legend()

        cov_path = os.path.join(base_dir, args.experiment_folder, 'cov_plot.png')
        plt.savefig(cov_path, dpi=150, bbox_inches='tight')
        print(f"Saved CoV plot to {cov_path}")
        plt.show()

    elif args.mode == "ensemble":
        # train a single VAE with an ensemble of decoders (shared encoder/prior)
        experiments_folder = os.path.join(base_dir, args.experiment_folder)
        os.makedirs(experiments_folder, exist_ok=True)

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)

        decoders = []
        for i in range(args.num_decoders):
            print(f"training decoder {i+1}/{args.num_decoders}")
            model.decoder = GaussianDecoder(new_decoder()).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, args.device)
            decoders.append(deepcopy(model.decoder))
            torch.save(model.decoder.state_dict(), f"{experiments_folder}/decoder_{i}.pt")

        torch.save(model.encoder.state_dict(), f"{experiments_folder}/encoder.pt")

        # compute ensemble geodesics
        model.eval()
        x = next(iter(mnist_test_loader))[0].to(device)
        z = model.encoder(x).mean.detach()

        pairs = torch.randperm(len(z))[:50].reshape(25, 2)
        geodesics = []
        for i in tqdm(range(25), desc="Computing ensemble geodesics"):
            a = z[pairs[i, 0]]
            b = z[pairs[i, 1]]
            geodesics.append(compute_ensemble_geodesic(a, b, decoders))

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        with torch.no_grad():
            all_x, all_y = [], []
            for batch_x, batch_y in mnist_test_loader:
                batch_z = model.encoder(batch_x.to(device)).mean
                all_x.append(batch_z.cpu())
                all_y.append(batch_y)
            all_z = torch.cat(all_x)
            all_labels = torch.cat(all_y)
        ax.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.3, s=10)
        for curve in geodesics:
            curve_np = curve.detach().cpu().numpy()
            ax.plot(curve_np[:, 0], curve_np[:, 1], 'r-', linewidth=1.5)
        plt.show()