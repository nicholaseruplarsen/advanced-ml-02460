# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def wl_hash(G, iterations=3):
    if G.number_of_nodes() == 0:
        return 'empty'
    return nx.weisfeiler_lehman_graph_hash(G, iterations=iterations)

def novelty_uniqueness(samples, train_graphs):
    train_hashes = set(wl_hash(g) for g in train_graphs)
    sample_hashes = [wl_hash(g) for g in samples]
    n = len(samples)
    novel_mask = [h not in train_hashes for h in sample_hashes]
    seen = {}
    unique_mask = []
    for h in sample_hashes:
        if h not in seen:
            seen[h] = True
            unique_mask.append(True)
        else:
            unique_mask.append(False)
    novel = sum(novel_mask) / n
    unique = sum(unique_mask) / n
    novel_unique = sum(a and b for a, b in zip(novel_mask, unique_mask)) / n
    return novel, unique, novel_unique

def graph_stats(G):
    if G.number_of_nodes() == 0:
        return [], [], []
    deg = [d for _, d in G.degree()]
    clust = list(nx.clustering(G).values())
    try:
        ec = list(nx.eigenvector_centrality_numpy(G, max_iter=1000).values())
    except Exception:
        ec = [0.0] * G.number_of_nodes()
    return deg, clust, ec

def collect_stats(graphs):
    deg_all, clust_all, ec_all = [], [], []
    for G in graphs:
        d, c, e = graph_stats(G)
        deg_all.extend(d); clust_all.extend(c); ec_all.extend(e)
    return np.array(deg_all), np.array(clust_all), np.array(ec_all)

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

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    rows = [('Empirical', deg_e, clust_e, ec_e),
            ('Baseline (ER)', deg_b, clust_b, ec_b),
            ('VGAE', deg_v, clust_v, ec_v)]
    cols = [('Node degree', deg_bins),
            ('Clustering coef.', clust_bins),
            ('Eigenvector centrality', ec_bins)]
    for i, (name, deg, clust, ec) in enumerate(rows):
        for j, (col_name, bins) in enumerate(cols):
            data = (deg, clust, ec)[j]
            ax = axes[i, j]
            ax.hist(data, bins=bins, density=True, color=['C0','C1','C2'][i], edgecolor='black', linewidth=0.4)
            if i == 0:
                ax.set_title(col_name)
            if j == 0:
                ax.set_ylabel(name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
