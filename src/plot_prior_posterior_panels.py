import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from flow import FlowPrior
from vae_bernoulli import BernoulliDecoder, GaussianEncoder, GaussianPrior, MoGPrior, VAE


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint(path_arg: str | None, candidates: list[str], name: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint for {name} not found: {p}")
        return p

    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p

    joined = ", ".join(candidates)
    raise FileNotFoundError(f"Checkpoint for {name} not found. Tried: {joined}")


def build_vae(prior_name: str, latent_dim: int, num_components: int, flow_layers: int, flow_hidden: int) -> VAE:
    if prior_name == "gaussian":
        prior = GaussianPrior(latent_dim)
    elif prior_name == "mog":
        prior = MoGPrior(latent_dim, num_components)
    elif prior_name == "flow":
        prior = FlowPrior(latent_dim, n_transformations=flow_layers, n_hidden=flow_hidden)
    else:
        raise ValueError(f"Unknown prior: {prior_name}")

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    return VAE(prior, BernoulliDecoder(decoder_net), GaussianEncoder(encoder_net))


def build_loader(batch_size: int):
    threshold = 0.5
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > threshold).float().squeeze()),
        ]
    )
    ds = datasets.MNIST("data/", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def collect_posterior(model: VAE, loader, device: str, n_samples: int):
    zs = []
    ys = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.encoder(x).sample().cpu()
            zs.append(z)
            ys.append(y)
            seen += x.size(0)
            if seen >= n_samples:
                break

    posterior = torch.cat(zs, dim=0)[:n_samples].numpy()
    labels = torch.cat(ys, dim=0)[:n_samples].numpy()
    return posterior, labels


def collect_prior(model: VAE, n_samples: int):
    model.eval()
    with torch.no_grad():
        prior = model.prior().sample(torch.Size([n_samples])).cpu().numpy()
    return prior


def tsne_project(posterior: np.ndarray, prior: np.ndarray, seed: int, perplexity: float):
    both = np.concatenate([posterior, prior], axis=0)
    valid_perplexity = min(perplexity, both.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        perplexity=valid_perplexity,
    )
    emb = tsne.fit_transform(both)
    n_post = posterior.shape[0]
    return emb[:n_post], emb[n_post:]


def draw_prior_contours(ax, prior_2d: np.ndarray, color: str = "royalblue", levels: int = 8):
    x = prior_2d[:, 0]
    y = prior_2d[:, 1]

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    pad_x = 0.10 * (x.max() - x.min() + 1e-8)
    pad_y = 0.10 * (y.max() - y.min() + 1e-8)
    grid_x = np.linspace(x.min() - pad_x, x.max() + pad_x, 200)
    grid_y = np.linspace(y.min() - pad_y, y.max() + pad_y, 200)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    density_at_samples = kde(xy)
    quantiles = [0.45, 0.58, 0.70, 0.80, 0.88, 0.94, 0.98]
    level_values = np.quantile(density_at_samples, quantiles)
    level_values = np.unique(level_values)
    if len(level_values) >= 2:
        ax.contour(xx, yy, zz, levels=level_values, colors="black", linewidths=1.2, alpha=0.55, zorder=3)


def main():
    parser = argparse.ArgumentParser(description="Make 3-panel t-SNE plots for posterior vs prior.")
    parser.add_argument("--device", type=str, default=default_device(), choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-components", type=int, default=10)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-hidden", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument(
        "--out",
        type=str,
        default="report/Figures/vae/prior_posterior_panels.pdf",
        help="Output image path",
    )
    parser.add_argument("--gaussian-ckpt", type=str, default=None)
    parser.add_argument("--mog-ckpt", type=str, default=None)
    parser.add_argument("--flow-ckpt", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_gauss = resolve_checkpoint(
        args.gaussian_ckpt,
        ["models_remote/vae_gaussian.pt", "models/vae_gaussian.pt", "vae_gaussian.pt"],
        "Gaussian prior",
    )
    ckpt_mog = resolve_checkpoint(
        args.mog_ckpt,
        ["models_remote/vae_mog.pt", "models/vae_mog.pt", "vae_mog.pt"],
        "MoG prior",
    )
    ckpt_flow = resolve_checkpoint(
        args.flow_ckpt,
        ["models_remote/vae_flow.pt", "models/vae_flow.pt", "vae_flow.pt"],
        "Flow prior",
    )

    loader = build_loader(args.batch_size)

    configs = [
        ("gaussian", "Gaussian Prior t-SNE", ckpt_gauss),
        ("mog", "MoG Prior t-SNE", ckpt_mog),
        ("flow", "Flow Prior t-SNE", ckpt_flow),
    ]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
    })

    # Vivid 10-class palette — more saturated than tab10
    vivid_colors = [
        "#3A86FF",  # 0 bright blue
        "#FF6B35",  # 1 vivid orange
        "#2DC653",  # 2 vivid green
        "#E63946",  # 3 vivid red
        "#9B5DE5",  # 4 purple
        "#8B4513",  # 5 brown
        "#F72585",  # 6 hot pink
        "#606060",  # 7 dark grey
        "#C5A800",  # 8 golden yellow
        "#00B4D8",  # 9 cyan
    ]
    cmap = ListedColormap(vivid_colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), 10)
    contour_color = "#1a1a2e"  # deep navy — dark and neutral, visible on any bg

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    scatter_for_colorbar = None

    for i, (prior_name, title, ckpt_path) in enumerate(configs):
        model = build_vae(
            prior_name,
            latent_dim=args.latent_dim,
            num_components=args.num_components,
            flow_layers=args.flow_layers,
            flow_hidden=args.flow_hidden,
        ).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.eval()

        posterior, labels = collect_posterior(model, loader, args.device, args.n_samples)
        prior = collect_prior(model, posterior.shape[0])

        posterior_2d, prior_2d = tsne_project(posterior, prior, args.seed + i, args.perplexity)

        ax = axes[i]
        scatter_for_colorbar = ax.scatter(
            posterior_2d[:, 0],
            posterior_2d[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.9,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        draw_prior_contours(ax, prior_2d, color=contour_color, levels=8)

        ax.set_title(title, fontsize=11, fontweight="bold", color="#111111")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2" if i == 0 else "", fontsize=8)
        ax.tick_params(labelsize=0, length=0)  # hide tick labels — unreadable at small scale

    # Single shared legend on first panel only
    legend_handles = [
        Line2D([0], [0], marker=".", linestyle="", markersize=6, color="#888888", label="Posterior"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Prior contour"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=8,
                   facecolor="white", edgecolor="#cccccc")

    cbar = fig.colorbar(scatter_for_colorbar, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    cbar.set_label("Digit", fontsize=9)
    cbar.set_ticks(range(10))
    cbar.ax.tick_params(labelsize=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
