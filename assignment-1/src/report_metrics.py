"""
Compute FID scores, sampling times, and latent DDPM prior/posterior plot.
"""
import time
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from vae_bernoulli import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder, BernoulliDecoder
from ddpm import DDPM, FcNetwork
from unet import Unet
from fid import compute_fid

DEVICE = "cuda"
M = 32
N = 1000  # samples for FID
CKPT = "models/mnist_classifier.pth"
OUT = "../report/Figures/summary.txt"

log = open(OUT, "w")

# ---------------------------------------------------------------------------
# Real images in [-1, 1]
# ---------------------------------------------------------------------------
real_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])),
    batch_size=256, shuffle=False)
x_real = torch.cat([x for x, _ in real_loader])[:N].to(DEVICE)


def build_bern_vae():
    enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M * 2))
    dec = nn.Sequential(nn.Linear(M, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 784), nn.Unflatten(-1, (28, 28)))
    return VAE(GaussianPrior(M), BernoulliDecoder(dec), GaussianEncoder(enc)).to(DEVICE)


def build_latent_vae(beta):
    enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M * 2))
    dec = nn.Sequential(nn.Linear(M, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 784), nn.Unflatten(-1, (28, 28)))
    return VAE(GaussianPrior(M), GaussianDecoder(dec), GaussianEncoder(enc), beta=beta).to(DEVICE)


# ---------------------------------------------------------------------------
# 1. VAE
# ---------------------------------------------------------------------------
vae = build_bern_vae()
vae.load_state_dict(torch.load("models/vae_gaussian.pt", map_location=DEVICE))
vae.eval()
with torch.no_grad():
    t0 = time.perf_counter()
    z = vae.prior().sample(torch.Size([N])).to(DEVICE)
    x_vae = vae.decoder(z).mean  # (N, 28, 28)
    t_vae = time.perf_counter() - t0
x_vae_fid = x_vae.unsqueeze(1).clamp(0, 1) * 2.0 - 1.0  # -> [-1, 1], (N, 1, 28, 28)
fid_vae = compute_fid(x_real, x_vae_fid, device=DEVICE, classifier_ckpt=CKPT)

# ---------------------------------------------------------------------------
# 2. DDPM (U-Net)
# ---------------------------------------------------------------------------
ddpm = DDPM(Unet(), T=1000).to(DEVICE)
ddpm.load_state_dict(torch.load("models/ddpm_mnist.pt", map_location=DEVICE))
ddpm.eval()
with torch.no_grad():
    t0 = time.perf_counter()
    x_ddpm = ddpm.sample((N, 784)).view(N, 1, 28, 28).clamp(-1, 1)
    t_ddpm = time.perf_counter() - t0
fid_ddpm = compute_fid(x_real, x_ddpm, device=DEVICE, classifier_ckpt=CKPT)

# ---------------------------------------------------------------------------
# 3. Latent DDPM — all beta values
# ---------------------------------------------------------------------------
betas = [(1e-6, "1e-6"), (1e-3, "1e-3"), (1e-1, "1e-1"), (1.0, "1.0")]
fid_latent, time_latent = {}, {}

for beta, beta_str in betas:
    vae_l = build_latent_vae(beta)
    vae_l.load_state_dict(torch.load(f"models/latent_vae_b{beta_str}.pt", map_location=DEVICE))
    vae_l.eval()
    ddpm_l = DDPM(FcNetwork(M, 256), T=1000).to(DEVICE)
    ddpm_l.load_state_dict(torch.load(f"models/latent_ddpm_b{beta_str}.pt", map_location=DEVICE))
    ddpm_l.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        z = ddpm_l.sample((N, M))
        x_g = vae_l.decoder(z).mean.unsqueeze(1).clamp(0, 1) * 2.0 - 1.0
        t_l = time.perf_counter() - t0
    fid_l = compute_fid(x_real, x_g, device=DEVICE, classifier_ckpt=CKPT)
    fid_latent[beta_str] = fid_l
    time_latent[beta_str] = t_l

# ---------------------------------------------------------------------------
# 4. Latent DDPM prior/posterior plot (beta=1e-6)
# ---------------------------------------------------------------------------
beta_p = 1e-6
beta_p_str = "1e-6"
vae_p = build_latent_vae(beta_p)
vae_p.load_state_dict(torch.load(f"models/latent_vae_b{beta_p_str}.pt", map_location=DEVICE))
vae_p.eval()
ddpm_p = DDPM(FcNetwork(M, 256), T=1000).to(DEVICE)
ddpm_p.load_state_dict(torch.load(f"models/latent_ddpm_b{beta_p_str}.pt", map_location=DEVICE))
ddpm_p.eval()

cont_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze()),
    ])),
    batch_size=256, shuffle=False)

post, lbls = [], []
with torch.no_grad():
    for x, y in cont_loader:
        post.append(vae_p.encoder(x.to(DEVICE)).mean.cpu())
        lbls.append(y)
post = torch.cat(post).numpy()
lbls = torch.cat(lbls).numpy()
nt = len(post)

with torch.no_grad():
    vae_prior_samps = vae_p.prior().sample(torch.Size([nt])).cpu().numpy()
    ddpm_latents = ddpm_p.sample((nt, M)).cpu().numpy()

all_2d = TSNE(n_components=2, random_state=42).fit_transform(
    np.concatenate([post, vae_prior_samps, ddpm_latents]))

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(all_2d[nt:2*nt, 0], all_2d[nt:2*nt, 1], s=1, alpha=0.15, color='lightgray', zorder=1)
ax.scatter(all_2d[2*nt:, 0], all_2d[2*nt:, 1], s=1, alpha=0.15, color='steelblue', zorder=2)
sc = ax.scatter(all_2d[:nt, 0], all_2d[:nt, 1], s=2, alpha=0.6, c=lbls, cmap='tab10', zorder=3)
plt.colorbar(sc, ax=ax, ticks=range(10), label='Digit class (aggregate posterior)')
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=9, label='β-VAE prior (Gaussian)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=9, label='Latent DDPM samples'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=9,
           label='Aggregate posterior\n(colored by digit class →)'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_title('β-VAE latent space (β=1e-6): prior, DDPM samples, and aggregate posterior')
plt.tight_layout()
plt.savefig('report/Figures/ddpm/latent_ddpm_prior_posterior.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log.write("========== FID & SAMPLING SPEED SUMMARY ==========\n\n")
log.write(f"{'Model':<30} {'FID':>8}  {'Sampling speed':>20}\n")
log.write("-" * 62 + "\n")
log.write(f"{'VAE (Gaussian)':<30} {fid_vae:>8.2f}  {N/t_vae:>16,.0f} samp/s\n")
log.write(f"{'DDPM (U-Net)':<30} {fid_ddpm:>8.2f}  {N/t_ddpm:>16,.0f} samp/s\n")
for beta, beta_str in betas:
    label = f"Latent DDPM b={beta_str}"
    log.write(f"{label:<30} {fid_latent[beta_str]:>8.2f}  {N/time_latent[beta_str]:>16,.0f} samp/s\n")
log.close()
