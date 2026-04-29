import numpy as np
import networkx as nx

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
    ec_by_node = {node: 0.0 for node in G.nodes()}
    for component in nx.connected_components(G):
        H = G.subgraph(component)
        if H.number_of_nodes() == 1:
            continue
        try:
            ec_by_node.update(nx.eigenvector_centrality_numpy(H).items())
        except Exception:
            ec_by_node.update(nx.eigenvector_centrality(H, max_iter=1000).items())
    ec = [ec_by_node[node] for node in G.nodes()]
    return deg, clust, ec

def collect_stats(graphs):
    deg_all, clust_all, ec_all = [], [], []
    for G in graphs:
        d, c, e = graph_stats(G)
        deg_all.extend(d); clust_all.extend(c); ec_all.extend(e)
    return np.array(deg_all), np.array(clust_all), np.array(ec_all)

