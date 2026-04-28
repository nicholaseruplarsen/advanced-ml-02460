# %%
import torch
import networkx as nx
from data import empirical_size_distribution, edge_density_per_size

def fit_baseline(train_graphs):
    sizes, probs = empirical_size_distribution(train_graphs)
    densities = edge_density_per_size(train_graphs)
    return sizes, probs, densities

def sample_baseline(num_samples, sizes, probs, densities, seed=0):
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_samples):
        idx = torch.multinomial(probs, 1, generator=g).item()
        n = int(sizes[idx].item())
        r = densities[n]
        # symmetric Bernoulli on upper triangle (mask AFTER thresholding,
        # otherwise the zeroed lower triangle satisfies 0 < r and floods the graph)
        u = torch.rand(n, n, generator=g)
        A = (u < r).float()
        A = torch.triu(A, diagonal=1)
        A = A + A.t()
        graphs.append(adj_to_nx(A))
    return graphs

def adj_to_nx(A):
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = torch.triu_indices(n, n, 1)
    edges = (A[rows, cols] > 0.5).nonzero(as_tuple=False).flatten().tolist()
    for e in edges:
        G.add_edge(int(rows[e]), int(cols[e]))
    return G
