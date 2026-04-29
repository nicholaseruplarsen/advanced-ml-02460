import math
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from data import load_mutag, NODE_FEATURE_DIM
from baseline import fit_baseline, sample_baseline, adj_to_nx
from model import VGAE
from metrics import novelty_uniqueness, collect_stats

device = 'cpu'
NUM_SAMPLES = 1000

dataset, train_dataset, val_dataset, test_dataset = load_mutag(root='../data')
train_graphs_pyg = list(train_dataset)

# convert training graphs to nx (for novelty + statistics reference)
train_graphs_nx = [to_networkx(d, to_undirected=True) for d in train_graphs_pyg]

sizes, probs, densities = fit_baseline(train_graphs_pyg)
baseline_graphs = sample_baseline(NUM_SAMPLES, sizes, probs, densities, seed=0)

ckpt = torch.load('assignment-3/vgae.pt', map_location=device, weights_only=False)
model = VGAE(node_feature_dim=NODE_FEATURE_DIM, state_dim=64, latent_dim=16, num_message_passing_rounds=4).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
logit_shift = -math.log(ckpt['pos_weight'])

torch.manual_seed(1)
vae_graphs = []
with torch.no_grad():
    for _ in range(NUM_SAMPLES):
        idx = torch.multinomial(probs, 1).item()
        n = int(sizes[idx].item())
        A = model.sample(n, device=device, logit_shift=logit_shift, target_density=densities[n])
        vae_graphs.append(adj_to_nx(A))

# Novelty / uniqueness table
b_novel, b_unique, b_nu = novelty_uniqueness(baseline_graphs, train_graphs_nx)
v_novel, v_unique, v_nu = novelty_uniqueness(vae_graphs, train_graphs_nx)

print()
print('|                       | Novel | Unique | Novel+unique |')
print('|:----------------------|:-----:|:------:|:------------:|')
print(f'| Baseline (Erdös-Rényi) | {b_novel*100:5.1f}% | {b_unique*100:5.1f}% | {b_nu*100:11.1f}% |')
print(f'| Deep generative model  | {v_novel*100:5.1f}% | {v_unique*100:5.1f}% | {v_nu*100:11.1f}% |')

emp_stats = collect_stats(train_graphs_nx)
base_stats = collect_stats(baseline_graphs)
vae_stats = collect_stats(vae_graphs)

METHOD_COLORS = {
    'Empirical': '#23395b',
    'Erdös-Rényi': '#c45a2a',
    'VGAE': '#2f7f6f',
}

def plot_histogram_grid(empirical, baseline, vae, save_path):
    deg_e, clust_e, ec_e = empirical
    deg_b, clust_b, ec_b = baseline
    deg_v, clust_v, ec_v = vae

    deg_max = max(deg_e.max() if len(deg_e) else 1, deg_b.max() if len(deg_b) else 1, deg_v.max() if len(deg_v) else 1)
    deg_bins = np.arange(0, int(deg_max) + 2) - 0.5
    clust_bins = np.linspace(0, 1, 21)
    ec_lo = min(np.min(x) if len(x) else 0 for x in [ec_e, ec_b, ec_v])
    ec_hi = max(np.max(x) if len(x) else 1 for x in [ec_e, ec_b, ec_v])
    if ec_hi <= ec_lo:
        ec_hi = ec_lo + 1.0
    ec_bins = np.linspace(ec_lo, ec_hi, 21)

    plt.rcdefaults()
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.2))
    rows = [('Empirical', deg_e, clust_e, ec_e),
            ('Erdös-Rényi', deg_b, clust_b, ec_b),
            ('VGAE', deg_v, clust_v, ec_v)]
    cols = [('Node degree', deg_bins),
            ('Clustering coef.', clust_bins),
            ('Eigenvector centrality', ec_bins)]
    for i, (name, deg, clust, ec) in enumerate(rows):
        for j, (col_name, bins) in enumerate(cols):
            data = (deg, clust, ec)[j]
            ax = axes[i, j]
            color = METHOD_COLORS[name]
            ax.hist(data, bins=bins, density=True, color=color, alpha=0.80, linewidth=0.65)
            ax.axvline(np.mean(data), linewidth=1.1, alpha=0.85)
            ax.grid(axis='y', linewidth=0.75, alpha=0.9)
            ax.grid(axis='x', visible=False)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#b8afa0')
            if i == 0:
                ax.set_title(col_name, fontsize=11, weight='bold')
            # if j == 0:
                # ax.set_ylabel(name, fontsize=11, weight='bold', color=color)
            else:
                ax.set_ylabel('')
            if i < 2:
                ax.set_xlabel('')
            ax.tick_params(labelsize=8)
    # fig.suptitle('MUTAG graph statistic distributions', fontsize=14, weight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

plot_histogram_grid(emp_stats, base_stats, vae_stats, 'assignment-3/report/figures/histograms.png')

def _representative_graphs(graphs, num_graphs=5):
    if len(graphs) <= num_graphs:
        return graphs
    scored = sorted(
        graphs,
        key=lambda g: (g.number_of_nodes(), g.number_of_edges())
    )
    positions = np.linspace(0, len(scored) - 1, num_graphs).round().astype(int)
    return [scored[i] for i in positions]

def plot_graph_gallery(empirical_graphs, baseline_graphs, vae_graphs, save_path, num_graphs=3):
    plt.rcdefaults()
    rows = [
        ('Empirical', _representative_graphs(empirical_graphs, num_graphs)),
        ('Erdös-Rényi', _representative_graphs(baseline_graphs, num_graphs)),
        ('VGAE', _representative_graphs(vae_graphs, num_graphs)),
    ]
    fig, axes = plt.subplots(3, num_graphs, figsize=(2.05 * num_graphs, 5.75))
    for i, (name, graphs) in enumerate(rows):
        for j, G in enumerate(graphs):
            ax = axes[i, j]
            color = METHOD_COLORS[name]
            degrees = np.array([d for _, d in G.degree()], dtype=float)
            if G.number_of_edges() > 0:
                pos = nx.spring_layout(G, seed=1000 + 17 * i + j, k=0.85, iterations=80)
            else:
                pos = nx.circular_layout(G)
            nx.draw_networkx_edges(G, pos, ax=ax, width=1.15, alpha=0.32, edge_color='#4b5563')
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=55 + 18 * degrees,
                node_color=color, edgecolors='white', linewidths=0.7, alpha=0.92
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=11, weight='bold', color=color, rotation=0, labelpad=42, va='center')
            ax.set_title(f'N={G.number_of_nodes()}, E={G.number_of_edges()}', fontsize=8, pad=1)
    # fig.suptitle('Representative graph samples', fontsize=14, weight='bold', y=0.99)
    fig.tight_layout(rect=[0.03, 0, 1, 0.965])
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

plot_graph_gallery(train_graphs_nx, baseline_graphs, vae_graphs, 'assignment-3/report/figures/graph_samples.png', num_graphs=3)
print('saved histograms.png and graph_samples.png')

from metrics import wl_hash
from collections import Counter

def wl_multiplicity(samples):
    # how many distinct hashes, max bucket size, entropy of hash distribution
    hashes = [wl_hash(g) for g in samples]
    counts = Counter(hashes)
    n = len(samples)
    n_distinct = len(counts)
    max_bucket = max(counts.values())
    probs = np.array(list(counts.values())) / n
    entropy = float(-(probs * np.log2(probs)).sum())
    return n_distinct, max_bucket, entropy

def degree_validity(graphs, max_valence=4):
    # fraction of nodes with degree > max_valence
    # MUTAG nodes are atoms, max valence ~4 (carbon).
    n_total = 0
    n_invalid = 0
    for G in graphs:
        for _, d in G.degree():
            n_total += 1
            if d > max_valence:
                n_invalid += 1
    return n_invalid / max(n_total, 1)

b_distinct, b_max, b_ent = wl_multiplicity(baseline_graphs)
v_distinct, v_max, v_ent = wl_multiplicity(vae_graphs)
e_distinct, e_max, e_ent = wl_multiplicity(train_graphs_nx)
b_inv = degree_validity(baseline_graphs)
v_inv = degree_validity(vae_graphs)
e_inv = degree_validity(train_graphs_nx)

print()
print('|                       | Distinct hashes | Max bucket | Entropy (bits) | Degree>4 |')
print('|:----------------------|:---------------:|:----------:|:--------------:|:--------:|')
print(f'| Empirical (100 train)  | {e_distinct:15d} | {e_max:10d} | {e_ent:14.2f} | {e_inv*100:7.1f}% |')
print(f'| Baseline (Erdös-Rényi) | {b_distinct:15d} | {b_max:10d} | {b_ent:14.2f} | {b_inv*100:7.1f}% |')
print(f'| Deep generative model  | {v_distinct:15d} | {v_max:10d} | {v_ent:14.2f} | {v_inv*100:7.1f}% |')