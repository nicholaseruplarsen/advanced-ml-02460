# %%
import torch
from torch_geometric.loader import DataLoader
from data import load_mutag, NODE_FEATURE_DIM
from model import VGAE, vgae_loss

device = 'cpu'

# %%
dataset, train_dataset, val_dataset, test_dataset = load_mutag()
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=44)

# %%
# Estimate pos_weight: training graphs are sparse, so positive edge class is rare.
# count num_real_pairs vs num_edges across training set
num_pos = 0
num_total = 0
for d in train_dataset:
    n = d.num_nodes
    num_pos += d.edge_index.shape[1]  # both directions count, that's fine because we compare to dense A which has both
    num_total += n * n - n  # exclude diagonal
pos_weight = torch.tensor((num_total - num_pos) / max(num_pos, 1), device=device)
print(f'pos_weight = {pos_weight.item():.2f}')

# %%
torch.manual_seed(0)
model = VGAE(node_feature_dim=NODE_FEATURE_DIM, state_dim=64, latent_dim=16, num_message_passing_rounds=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# %%
epochs = 500
beta_max = 0.1
beta_warmup = 100

train_recons, train_kls, val_recons, val_kls = [], [], [], []

for epoch in range(epochs):
    beta = beta_max * min(1.0, (epoch + 1) / beta_warmup)
    model.train()
    tr_recon = 0.0
    tr_kl = 0.0
    n_train = 0
    for data in train_loader:
        data = data.to(device)
        logits, A, mask, mu_d, logvar_d = model(data.x, data.edge_index, data.batch)
        recon, kl = vgae_loss(logits, A, mask, mu_d, logvar_d, pos_weight=pos_weight)
        loss = recon + beta * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = int(data.batch.max().item()) + 1
        tr_recon += recon.detach().item() * bs
        tr_kl += kl.detach().item() * bs
        n_train += bs
    tr_recon /= n_train
    tr_kl /= n_train
    scheduler.step()

    model.eval()
    with torch.no_grad():
        v_recon = 0.0
        v_kl = 0.0
        n_val = 0
        for data in val_loader:
            data = data.to(device)
            logits, A, mask, mu_d, logvar_d = model(data.x, data.edge_index, data.batch)
            recon, kl = vgae_loss(logits, A, mask, mu_d, logvar_d, pos_weight=pos_weight)
            bs = int(data.batch.max().item()) + 1
            v_recon += recon.item() * bs
            v_kl += kl.item() * bs
            n_val += bs
        v_recon /= n_val
        v_kl /= n_val

    train_recons.append(tr_recon); train_kls.append(tr_kl)
    val_recons.append(v_recon); val_kls.append(v_kl)

    if (epoch + 1) % 25 == 0:
        print(f'epoch {epoch+1:4d} | beta={beta:.3f} | train recon={tr_recon:.3f} kl={tr_kl:.3f} | val recon={v_recon:.3f} kl={v_kl:.3f}')

# %%
torch.save({
    'state_dict': model.state_dict(),
    'train_recons': train_recons, 'train_kls': train_kls,
    'val_recons': val_recons, 'val_kls': val_kls,
    'pos_weight': pos_weight.item(),
}, 'vgae.pt')
print('saved vgae.pt')
