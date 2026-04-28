# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, to_dense_batch

NODE_FEATURE_DIM = 7

def load_mutag(root='./data/'):
    dataset = TUDataset(root=root, name='MUTAG')
    rng = torch.Generator().manual_seed(0)
    train, val, test = random_split(dataset, (100, 44, 44), generator=rng)
    return dataset, train, val, test

def graph_to_dense(data, max_nodes):
    # single graph -> (X [max_nodes, 7], A [max_nodes, max_nodes], mask [max_nodes])
    n = data.num_nodes
    X = torch.zeros(max_nodes, NODE_FEATURE_DIM)
    X[:n] = data.x
    A = torch.zeros(max_nodes, max_nodes)
    ei = data.edge_index
    A[ei[0], ei[1]] = 1.0
    mask = torch.zeros(max_nodes, dtype=torch.bool)
    mask[:n] = True
    return X, A, mask

def collate_dense(batch, max_nodes):
    Xs, As, masks = [], [], []
    for d in batch:
        X, A, m = graph_to_dense(d, max_nodes)
        Xs.append(X); As.append(A); masks.append(m)
    return torch.stack(Xs), torch.stack(As), torch.stack(masks)

def empirical_size_distribution(graphs):
    # returns (sizes [num_unique], probs [num_unique])
    counts = {}
    for g in graphs:
        n = g.num_nodes
        counts[n] = counts.get(n, 0) + 1
    sizes = sorted(counts.keys())
    total = sum(counts.values())
    probs = torch.tensor([counts[n] / total for n in sizes])
    return torch.tensor(sizes), probs

def edge_density_per_size(graphs):
    # returns dict {N: r} computed over training graphs of size N
    by_n = {}
    for g in graphs:
        n = g.num_nodes
        # PyG edge_index stores both directions, so num undirected edges = E/2
        e = g.edge_index.shape[1] // 2
        if n not in by_n:
            by_n[n] = [0, 0]
        by_n[n][0] += e
        by_n[n][1] += n * (n - 1) // 2
    return {n: (s[0] / s[1] if s[1] > 0 else 0.0) for n, s in by_n.items()}

def max_nodes(graphs):
    return max(g.num_nodes for g in graphs)
