"""
Latent DDPM: Train a DDPM in the latent space of a β-VAE with Gaussian likelihood.

Workflow:
1. Train a β-VAE (Gaussian decoder) on continuous MNIST
2. Encode training data into latent space
3. Train a DDPM (FcNetwork) on the latent vectors
4. Sample: DDPM generates latents → β-VAE decoder produces images
"""

import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from vae_bernoulli import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder, train as train_vae
from ddpm import DDPM, FcNetwork, train as train_ddpm


def encode_dataset(vae, data_loader, device):
    """Encode an entire dataset into latent space using the VAE encoder."""
    vae.eval()
    latents = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            q = vae.encoder(x)
            z = q.mean  # use mean, not sample, for stable training
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train_vae', 'train_ddpm', 'sample'])
    parser.add_argument('--vae-model', type=str, default='latent_vae.pt')
    parser.add_argument('--ddpm-model', type=str, default='latent_ddpm.pt')
    parser.add_argument('--samples', type=str, default='latent_ddpm_samples.png')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--vae-epochs', type=int, default=10)
    parser.add_argument('--ddpm-epochs', type=int, default=50)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--beta', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ddpm-T', type=int, default=1000)
    parser.add_argument('--ddpm-hidden', type=int, default=256)

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    M = args.latent_dim

    # Continuous MNIST scaled to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze()),
    ])
    mnist_train = datasets.MNIST('data/', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    # Build the β-VAE
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )
    prior = GaussianPrior(M)
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    vae = VAE(prior, decoder, encoder, beta=args.beta).to(device)

    if args.mode == 'train_vae':
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
        train_vae(vae, optimizer, train_loader, args.vae_epochs, device)
        torch.save(vae.state_dict(), args.vae_model)
        print(f'β-VAE saved to {args.vae_model}')

    elif args.mode == 'train_ddpm':
        # Load trained β-VAE
        vae.load_state_dict(torch.load(args.vae_model, map_location=device))
        vae.eval()

        # Encode training data into latent space
        print('Encoding training data...')
        latents = encode_dataset(vae, train_loader, device)
        print(f'Latent dataset shape: {latents.shape}')

        # Train DDPM on latent vectors
        latent_loader = torch.utils.data.DataLoader(latents, batch_size=args.batch_size, shuffle=True)

        network = FcNetwork(M, args.ddpm_hidden)
        ddpm = DDPM(network, T=args.ddpm_T).to(device)
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
        train_ddpm(ddpm, optimizer, latent_loader, args.ddpm_epochs, device)
        torch.save(ddpm.state_dict(), args.ddpm_model)
        print(f'Latent DDPM saved to {args.ddpm_model}')

    elif args.mode == 'sample':
        # Load both models
        vae.load_state_dict(torch.load(args.vae_model, map_location=device))
        vae.eval()

        network = FcNetwork(M, args.ddpm_hidden)
        ddpm = DDPM(network, T=args.ddpm_T).to(device)
        ddpm.load_state_dict(torch.load(args.ddpm_model, map_location=device))
        ddpm.eval()

        # Sample latents from DDPM, decode with VAE
        with torch.no_grad():
            z = ddpm.sample((64, M))
            samples = vae.decoder(z).mean.cpu()  # use decoder mean
            samples = samples.clamp(0, 1)
            save_image(samples.view(64, 1, 28, 28), args.samples)
        print(f'Samples saved to {args.samples}')
