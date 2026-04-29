import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_dense_adj

class GNNEncoder(nn.Module):
    def __init__(self, node_feature_dim, state_dim, latent_dim, num_message_passing_rounds):
        super().__init__()
        self.state_dim = state_dim
        self.num_rounds = num_message_passing_rounds

        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
        )
        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])
        self.mu_head = nn.Linear(state_dim, latent_dim)
        self.logvar_head = nn.Linear(state_dim, latent_dim)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        state = self.input_net(x)
        for r in range(self.num_rounds):
            message = self.message_net[r](state)
            agg = x.new_zeros((num_nodes, self.state_dim))
            agg = agg.index_add(0, edge_index[1], message[edge_index[0]])
            state = state + self.update_net[r](agg)
        mu = self.mu_head(state)
        logvar = self.logvar_head(state)
        return mu, logvar


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_dense):
        return z_dense @ z_dense.transpose(1, 2) + self.bias


class VGAE(nn.Module):
    def __init__(self, node_feature_dim=7, state_dim=64, latent_dim=16, num_message_passing_rounds=4):
        super().__init__()
        self.encoder = GNNEncoder(node_feature_dim, state_dim, latent_dim, num_message_passing_rounds)
        self.decoder = InnerProductDecoder()
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, batch):
        mu_n, logvar_n = self.encoder(x, edge_index)
        z_n = self.reparameterize(mu_n, logvar_n)
        z_dense, mask = to_dense_batch(z_n, batch)
        mu_dense, _ = to_dense_batch(mu_n, batch)
        logvar_dense, _ = to_dense_batch(logvar_n, batch)
        A = to_dense_adj(edge_index, batch, max_num_nodes=mask.shape[1])
        logits = self.decoder(z_dense)
        return logits, A, mask, mu_dense, logvar_dense

    def sample(self, n_nodes, device='cpu', logit_shift=0.0, target_density=None):
        z = torch.randn(1, n_nodes, self.latent_dim, device=device)
        return self._decode_bernoulli(z, logit_shift, target_density)

    def sample_conditional(self, x, edge_index, logit_shift=0.0):
        # Encode a real graph and sample a new graph from its node-level posterior.
        # The pos_weight in training balances BCE so the decoder bias settles at
        # sigmoid(b) ~ 0.5; logit_shift = -log(pos_weight) at sample time pulls
        # the average edge probability back to the empirical density q.
        with torch.no_grad():
            mu, logvar = self.encoder(x, edge_index)
            z = self.reparameterize(mu, logvar).unsqueeze(0)
            return self._decode_bernoulli(z, logit_shift)

    def _decode_bernoulli(self, z, logit_shift=0.0, target_density=None):
        logits = self.decoder(z)[0] + logit_shift
        if target_density is not None:
            logits = logits + density_calibration_shift(logits, target_density)
        probs = torch.sigmoid(logits)
        A = (torch.rand_like(probs) < probs).float()
        A = torch.triu(A, diagonal=1)
        return A + A.t()


def density_calibration_shift(logits, target_density, steps=30):
    # Corrects the global edge rate while preserving pairwise logit ordering.
    n = logits.shape[0]
    rows, cols = torch.triu_indices(n, n, 1, device=logits.device)
    edge_logits = logits[rows, cols]
    target = torch.as_tensor(target_density, device=logits.device).clamp(1e-4, 1 - 1e-4)
    lo = edge_logits.new_tensor(-20.0)
    hi = edge_logits.new_tensor(20.0)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if torch.sigmoid(edge_logits + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def vgae_loss(logits, A, mask, mu_dense, logvar_dense, pos_weight):
    B, N, _ = logits.shape
    pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).float()
    eye = torch.eye(N, device=A.device).unsqueeze(0)
    pair_mask = pair_mask * (1 - eye)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, A, pos_weight=pos_weight, reduction='none'
    )
    recon = (bce * pair_mask).sum() / pair_mask.sum()
    node_mask = mask.unsqueeze(-1).float()
    kl_per = -0.5 * (1 + logvar_dense - mu_dense.pow(2) - logvar_dense.exp())
    kl = (kl_per * node_mask).sum() / mask.sum()
    return recon, kl
