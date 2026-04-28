# %%
import math
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from data import load_mutag, NODE_FEATURE_DIM
from baseline import fit_baseline, sample_baseline, adj_to_nx
from model import VGAE
from metrics import novelty_uniqueness, collect_stats, plot_histogram_grid

device = 'cpu'
NUM_SAMPLES = 1000

# %%
dataset, train_dataset, val_dataset, test_dataset = load_mutag()
train_graphs_pyg = list(train_dataset)

# convert training graphs to nx (for novelty + statistics reference)
train_graphs_nx = [to_networkx(d, to_undirected=True) for d in train_graphs_pyg]

# %% Baseline
sizes, probs, densities = fit_baseline(train_graphs_pyg)
baseline_graphs = sample_baseline(NUM_SAMPLES, sizes, probs, densities, seed=0)

# %% VGAE
ckpt = torch.load('vgae.pt', map_location=device, weights_only=False)
model = VGAE(node_feature_dim=NODE_FEATURE_DIM, state_dim=64, latent_dim=16, num_message_passing_rounds=4).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Post-hoc bias calibration: with BCE pos_weight = (1-q)/q the decoder bias
# converges to sigmoid(b) ~ 0.5; subtracting log(pos_weight) from sample-time
# logits brings the average edge probability back to the empirical density q.
logit_shift = -math.log(ckpt['pos_weight'])

# Group training graphs by size so we can pick a posterior to sample from.
train_by_size = {}
for d in train_graphs_pyg:
    train_by_size.setdefault(d.num_nodes, []).append(d)

torch.manual_seed(1)
vae_graphs = []
with torch.no_grad():
    for _ in range(NUM_SAMPLES):
        idx = torch.multinomial(probs, 1).item()
        n = int(sizes[idx].item())
        candidates = train_by_size[n]
        pick = candidates[torch.randint(0, len(candidates), (1,)).item()]
        A = model.sample_conditional(pick.x.to(device), pick.edge_index.to(device), logit_shift=logit_shift)
        vae_graphs.append(adj_to_nx(A))

# %% Novelty / uniqueness table
b_novel, b_unique, b_nu = novelty_uniqueness(baseline_graphs, train_graphs_nx)
v_novel, v_unique, v_nu = novelty_uniqueness(vae_graphs, train_graphs_nx)

print()
print('|                       | Novel | Unique | Novel+unique |')
print('|:----------------------|:-----:|:------:|:------------:|')
print(f'| Baseline              | {b_novel*100:5.1f}% | {b_unique*100:5.1f}% | {b_nu*100:11.1f}% |')
print(f'| Deep generative model | {v_novel*100:5.1f}% | {v_unique*100:5.1f}% | {v_nu*100:11.1f}% |')

with open('results_table.md', 'w') as f:
    f.write('|                       | Novel | Unique | Novel+unique |\n')
    f.write('|:----------------------|:-----:|:------:|:------------:|\n')
    f.write(f'| Baseline              | {b_novel*100:5.1f}% | {b_unique*100:5.1f}% | {b_nu*100:11.1f}% |\n')
    f.write(f'| Deep generative model | {v_novel*100:5.1f}% | {v_unique*100:5.1f}% | {v_nu*100:11.1f}% |\n')

# %% Graph statistics histograms
emp_stats = collect_stats(train_graphs_nx)
base_stats = collect_stats(baseline_graphs)
vae_stats = collect_stats(vae_graphs)
plot_histogram_grid(emp_stats, base_stats, vae_stats, 'histograms.png')
print('saved histograms.png and results_table.md')
