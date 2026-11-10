
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple
import itertools
import time

class BasisGenerator:
    def __init__(self, mode='focal', s_coords=None):
        """
        mode: 'linear', 'gau1', 'gau2', 'cos1', 'cos2'
        s_coords: reference coords (N,2) for quantiles
        """
        self.mode = mode.lower()
        self.L = 2

        if s_coords is None:
            s_coords = torch.rand(1000, 2)

        self.mean = torch.mean(s_coords, dim=0, keepdim=True)
        self.std = torch.std(s_coords, dim=0, keepdim=True) + 1e-8
        s_norm = (s_coords - self.mean) / self.std
        abs_s = torch.abs(s_norm)

        self.psi_40 = torch.tensor([
            torch.quantile(abs_s[:, 0], 0.4),
            torch.quantile(abs_s[:, 1], 0.4)
        ])
        self.psi_60 = torch.tensor([
            torch.quantile(abs_s[:, 0], 0.6),
            torch.quantile(abs_s[:, 1], 0.6)
        ])

        s1 = s_norm[:, 0]
        s2 = s_norm[:, 1]

        h1_gau1 = torch.exp(-(s1 ** 2) / (2 * self.psi_40[0] ** 2))
        h2_gau1 = torch.exp(-(s2 ** 2) / (2 * self.psi_40[1] ** 2))
        self.gau1_mean = torch.stack([h1_gau1.mean(), h2_gau1.mean()])
        self.gau1_std  = torch.stack([h1_gau1.std() + 1e-8, h2_gau1.std() + 1e-8])

        h1_gau2 = torch.exp(-(s1 ** 2) / (2 * self.psi_60[0] ** 2))
        h2_gau2 = torch.exp(-(s2 ** 2) / (2 * self.psi_60[1] ** 2))
        self.gau2_mean = torch.stack([h1_gau2.mean(), h2_gau2.mean()])
        self.gau2_std  = torch.stack([h1_gau2.std() + 1e-8, h2_gau2.std() + 1e-8])

        print(f"Basis Init [{self.mode}] psi40={self.psi_40.numpy()} psi60={self.psi_60.numpy()}")
        print(f"gau1 mean/std: {self.gau1_mean.numpy()} / {self.gau1_std.numpy()}")
        print(f"gau2 mean/std: {self.gau2_mean.numpy()} / {self.gau2_std.numpy()}")

    def compute_h(self, s, device='cpu'):
        s = s.to(device)

        s_mean = torch.mean(s, dim=0, keepdim=True)
        s_std = torch.std(s, dim=0, keepdim=True) + 1e-8
        s_norm = (s - s_mean) / s_std

        s1 = s_norm[:, 0]
        s2 = s_norm[:, 1]

        psi40 = self.psi_40.to(device)
        psi60 = self.psi_60.to(device)

        if self.mode == 'linear':
            return s_norm

        elif self.mode == 'gau1':
            h1 = torch.exp(-(s1 ** 2) / (2 * psi40[0] ** 2))
            h2 = torch.exp(-(s2 ** 2) / (2 * psi40[1] ** 2))
            h  = torch.stack([h1, h2], dim=1)
            h = (h - self.gau1_mean.to(device)) / self.gau1_std.to(device)
            return h

        elif self.mode == 'gau2':
            h1 = torch.exp(-(s1 ** 2) / (2 * psi60[0] ** 2))
            h2 = torch.exp(-(s2 ** 2) / (2 * psi60[1] ** 2))
            h  = torch.stack([h1, h2], dim=1)
            h = (h - self.gau2_mean.to(device)) / self.gau2_std.to(device)
            return h

        elif self.mode == 'cos1':
            h1 = torch.cos(2 * np.pi * s1 / psi40[0])
            h2 = torch.cos(2 * np.pi * s2 / psi40[1])
            return torch.stack([h1, h2], dim=1)

        elif self.mode == 'cos2':
            h1 = torch.cos(2 * np.pi * s1 / psi60[0])
            h2 = torch.cos(2 * np.pi * s2 / psi60[1])
            return torch.stack([h1, h2], dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def nb_log_prob_matrix(mu, psi, y):
    """
    Return log_prob with shape (N, G).
    psi: (1,G) broadcast, mu: (N,G), y: (N,G)
    """
    denom = psi + mu + 1e-8
    probs = psi / denom
    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    dist = torch.distributions.NegativeBinomial(total_count=psi, probs=probs)
    return dist.log_prob(y)


def nb_loss_mean(model, data_batch):
    """
    Training loss: mean negative log-likelihood over (N*G).
    """
    y = data_batch['y']
    s = data_batch['s']
    omega = data_batch['omega']

    mu = model(s, omega)
    psi = model.get_psi()
    logp = nb_log_prob_matrix(mu, psi, y)
    nll = -logp.mean()
    return nll


def nb_nll_per_spot(model, data_batch):
    """
    Inference helper: per-spot negative log-likelihood, shape (N,)
    SUM over genes (NOT mean).
    """
    y = data_batch['y']
    s = data_batch['s']
    omega = data_batch['omega']

    mu = model(s, omega)
    psi = model.get_psi()
    logp = nb_log_prob_matrix(mu, psi, y)
    nll_per_spot = -logp.sum(dim=1)
    return nll_per_spot


def soft_threshold_adaptive(w, base_lambda, v_hat, epsilon=1e-8):
    adaptive_threshold = base_lambda / (torch.sqrt(v_hat) + epsilon)
    return torch.sign(w) * torch.maximum(torch.abs(w) - adaptive_threshold, torch.zeros_like(w))


class CTSVModel(nn.Module):
    def __init__(self, n_cell_types, n_genes, basis_gen):
        super().__init__()
        self.K = n_cell_types
        self.G = n_genes
        self.basis_gen = basis_gen
        self.L = basis_gen.L

        self.theta = nn.Parameter(torch.zeros(1, self.K, self.G))
        self.beta = nn.Parameter(torch.zeros(self.L, self.K, self.G))
        self.log_psi = nn.Parameter(torch.zeros(1, self.G))

    def forward(self, s, omega):
        h = self.basis_gen.compute_h(s, device=s.device)
        spatial_part = torch.einsum('nl,lkg->nkg', h, self.beta)
        eta = self.theta + spatial_part
        log_mu = torch.sum(omega.unsqueeze(-1) * eta, dim=1)
        return torch.exp(log_mu)

    def get_psi(self):
        return torch.exp(self.log_psi)


class FederatedClient:
    def __init__(self, client_id, data, n_cell_types, n_genes, basis_gen,
                 l1_lambda, device='cpu', missing_cell_types: Optional[List[int]] = None):
        self.id = int(client_id)
        self.data = {k: v.to(device) for k, v in data.items()}
        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.basis_gen = basis_gen
        self.device = device

        self.E = 10
        self.lr = 0.05
        self.l1_lambda = l1_lambda
        self.alpha = 0.1

        self.missing_cell_types = list(missing_cell_types) if missing_cell_types else []

    def _enforce_missing_omega_zero_and_renorm(self):
        """
        Double-safety: for missing cell types, force omega[:,k]=0 and renormalize others.
        """
        missing = self.missing_cell_types
        if not missing:
            return

        with torch.no_grad():
            self.data["omega"][:, missing] = 0.0
            keep_cols = [k for k in range(self.n_cell_types) if k not in missing]
            if len(keep_cols) > 0:
                row_sum = self.data["omega"][:, keep_cols].sum(dim=1, keepdim=True)
                row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
                self.data["omega"][:, keep_cols] = self.data["omega"][:, keep_cols] / row_sum
            self.data["omega"][:, missing] = 0.0

    def local_update(self, global_model_state, global_delta_G):
        """
        Local training with Adam-like updates + drift + L1 prox on beta only.
        Missing cell types: freeze theta/beta slices for those types (grad+drift masked, and hard-freeze).
        """
        self._enforce_missing_omega_zero_and_renorm()

        model = CTSVModel(self.n_cell_types, self.n_genes, self.basis_gen).to(self.device)
        model.load_state_dict(global_model_state)
        model.train()

        named_params = list(model.named_parameters())
        params = [p for _, p in named_params]

        missing = self.missing_cell_types
        if missing:
            theta_fixed = model.theta.data[:, missing, :].clone()
            beta_fixed  = model.beta.data[:, missing, :].clone()
        else:
            theta_fixed = None
            beta_fixed = None

        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) + 1e-8 for p in params]

        current_lr = float(self.lr)
        beta1, beta2 = 0.9, 0.99
        eps = 1e-8

        for epoch in range(self.E):
            loss = nb_loss_mean(model, self.data)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for i, (pname, p) in enumerate(named_params):
                    if p.grad is None:
                        continue

                    grad = p.grad

                    drift = 0.0
                    if global_delta_G is not None:
                        drift = self.alpha * global_delta_G[i]

                    if missing and pname in ("theta", "beta"):
                        if pname == "theta":
                            grad[:, missing, :] = 0.0
                            if isinstance(drift, torch.Tensor):
                                drift[:, missing, :] = 0.0
                        else:
                            grad[:, missing, :] = 0.0
                            if isinstance(drift, torch.Tensor):
                                drift[:, missing, :] = 0.0

                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grad ** 2)

                    m_hat = m[i] / (1.0 - beta1 ** (epoch + 1))
                    v_hat = v[i] / (1.0 - beta2 ** (epoch + 1))

                    step = m_hat / (torch.sqrt(v_hat) + eps)
                    p_temp = p.data - current_lr * (step + drift)

                    if pname == "beta":
                        base_thresh = current_lr * self.l1_lambda
                        p.data = soft_threshold_adaptive(p_temp, base_thresh, v_hat, eps)
                    else:
                        p.data = p_temp

                if missing:
                    model.theta.data[:, missing, :] = theta_fixed
                    model.beta.data[:, missing, :] = beta_fixed

            current_lr *= 0.98

        delta = {}
        state = model.state_dict()
        for k in global_model_state:
            delta[k] = state[k] - global_model_state[k]
        return delta, float(loss.item())

    def compute_empirical_fisher_diag_beta(self, global_model_state,
                                          fisher_spot_sample=None,
                                          seed=1234):
        """
        Empirical Fisher DIAG for beta:
            Fisher_diag ≈ Σ_i (∂ℓ_i/∂β)^2
        where ℓ_i is per-spot NLL with sum over genes.
        Missing cell types: enforce omega[:,missing]=0 and zero out fisher slice beta[:,missing,:].
        """
        self._enforce_missing_omega_zero_and_renorm()

        model = CTSVModel(self.n_cell_types, self.n_genes, self.basis_gen).to(self.device)
        model.load_state_dict(global_model_state)
        model.eval()

        nll_per_spot = nb_nll_per_spot(model, self.data)
        N = int(nll_per_spot.shape[0])

        if fisher_spot_sample is None or fisher_spot_sample >= N:
            idx = torch.arange(N, device=self.device)
            scale = 1.0
        else:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed + self.id)
            perm = torch.randperm(N, generator=g, device=self.device)
            idx = perm[:int(fisher_spot_sample)]
            scale = float(N) / float(fisher_spot_sample)

        fisher_diag = torch.zeros_like(model.beta)
        missing = self.missing_cell_types

        for i in idx:
            gi = torch.autograd.grad(nll_per_spot[i], model.beta, retain_graph=True)[0]
            if missing:
                gi[:, missing, :] = 0.0
            fisher_diag += gi.pow(2)

        fisher_diag = fisher_diag * scale
        if missing:
            fisher_diag[:, missing, :] = 0.0

        return fisher_diag.detach().cpu().numpy()


def load_clients_from_combined_csv(
    y_csv_path: str,
    omega_s_csv_path: str,
    G: int,
    K: int,
    device: str = "cpu",
    gene_prefix: str = "Gene_",
    omega_prefix: str = "Omega_Cell_",
    missing_celltypes_by_client: Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """
    Read:
      - AllClients_Y.csv: columns = [Client_ID, Spot_ID, <G gene columns>]
      - Input_Omega_S.csv: columns = [Client_ID, Spot_ID, Coord_X, Coord_Y, <omega cols possibly missing some>]

    Behavior:
      - merges on (Client_ID, Spot_ID)
      - pads omega to global K with zeros if some omega columns are missing
      - for clients in missing_celltypes_by_client: enforces omega[:,missing]=0 and renormalizes remaining columns
      - DOES NOT clip omega to 1e-8 globally (to preserve true zeros)
    """
    df_y = pd.read_csv(y_csv_path)
    df_o = pd.read_csv(omega_s_csv_path)

    for col in ["Client_ID", "Spot_ID"]:
        if col not in df_y.columns:
            raise ValueError(f"AllClients_Y missing column: {col}")
        if col not in df_o.columns:
            raise ValueError(f"Input_Omega_S missing column: {col}")

    coord_cols = ["Coord_X", "Coord_Y"]
    for col in coord_cols:
        if col not in df_o.columns:
            raise ValueError(f"Input_Omega_S missing column: {col}")

    gene_cols = [f"{gene_prefix}{i}" for i in range(G)]
    if not all(c in df_y.columns for c in gene_cols):
        candidate = [c for c in df_y.columns if c not in ["Client_ID", "Spot_ID"]]
        if len(candidate) != G:
            raise ValueError(
                f"Cannot determine gene columns: expected {G} gene cols, found {len(candidate)}"
            )
        gene_cols = candidate

    omega_cols_present = []
    omega_idx_present = []
    for c in df_o.columns:
        if c.startswith(omega_prefix):
            try:
                idx = int(c.replace(omega_prefix, ""))
                omega_cols_present.append(c)
                omega_idx_present.append(idx)
            except Exception:
                pass

    if len(omega_cols_present) == 0:
        candidate = [c for c in df_o.columns if c not in ["Client_ID", "Spot_ID", "Coord_X", "Coord_Y"]]
        omega_cols_present = candidate
        omega_idx_present = list(range(len(candidate)))

    df = df_y[["Client_ID", "Spot_ID"] + gene_cols].merge(
        df_o[["Client_ID", "Spot_ID"] + coord_cols + omega_cols_present],
        on=["Client_ID", "Spot_ID"],
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(["Client_ID", "Spot_ID"]).reset_index(drop=True)

    client_ids = sorted(df["Client_ID"].unique().tolist())
    clients_data = []

    if missing_celltypes_by_client is None:
        missing_celltypes_by_client = {}

    for cid in client_ids:
        sub = df[df["Client_ID"] == cid]

        Y = sub[gene_cols].to_numpy(dtype=np.float32)
        S = sub[coord_cols].to_numpy(dtype=np.float32)
        Omega_present = sub[omega_cols_present].to_numpy(dtype=np.float32)

        Omega = np.zeros((Omega_present.shape[0], K), dtype=np.float32)
        for j, k_idx in enumerate(omega_idx_present):
            if 0 <= k_idx < K:
                Omega[:, k_idx] = Omega_present[:, j]

        missing_k = set(missing_celltypes_by_client.get(int(cid), []))
        for k in missing_k:
            if 0 <= k < K:
                Omega[:, k] = 0.0

        keep_cols = [k for k in range(K) if k not in missing_k]
        if len(keep_cols) > 0:
            row_sum = Omega[:, keep_cols].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            Omega[:, keep_cols] = Omega[:, keep_cols] / row_sum
        for k in missing_k:
            Omega[:, k] = 0.0

        data = {
            "y": torch.tensor(Y, dtype=torch.float32, device=device),
            "s": torch.tensor(S, dtype=torch.float32, device=device),
            "omega": torch.tensor(Omega, dtype=torch.float32, device=device),
        }
        clients_data.append(data)

    return clients_data, client_ids


def load_global_true_beta(
    beta_csv_path: str,
    G: int,
    K: int,
    L_dim: int = 2,
    feature_to_l: Optional[Dict[str, int]] = None,
    device: str = "cpu",
):
    """
    Read GlobalTrue_Beta.csv with columns:
      Gene_ID, Cell_Type, Feature, True_Beta
    Feature is 'h1' or 'h2' (mapping to l=0,1).
    Return: (L_dim, K, G)
    """
    if feature_to_l is None:
        feature_to_l = {"h1": 0, "h2": 1}

    df = pd.read_csv(beta_csv_path)
    required_cols = ["Gene_ID", "Cell_Type", "Feature", "True_Beta"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{beta_csv_path} missing column: {c}")

    if df.shape[0] != G * K * L_dim:
        raise ValueError(
            f"Row count mismatch: expected {G*K*L_dim}, got {df.shape[0]} in {beta_csv_path}"
        )

    beta_np = np.zeros((L_dim, K, G), dtype=np.float32)
    for _, row in df.iterrows():
        g = int(row["Gene_ID"])
        k = int(row["Cell_Type"])
        feat = str(row["Feature"])
        l = feature_to_l.get(feat, None)
        if l is None:
            raise ValueError(f"Unknown Feature='{feat}' in {beta_csv_path} (expect h1/h2)")
        beta_np[l, k, g] = float(row["True_Beta"])

    return torch.tensor(beta_np, dtype=torch.float32, device=device)


def build_active_matrix(clients: List[FederatedClient], K: int, device) -> torch.Tensor:
    """
    active[m,k] = 1 if client m has block k, else 0.
    """
    M = len(clients)
    active = torch.ones((M, K), dtype=torch.float32, device=device)
    for m, c in enumerate(clients):
        for k_miss in getattr(c, "missing_cell_types", []):
            if 0 <= int(k_miss) < K:
                active[m, int(k_miss)] = 0.0
    return active


def aggregate_deltas_blockwise(
    client_deltas: List[Dict[str, torch.Tensor]],
    global_state: Dict[str, torch.Tensor],
    active: torch.Tensor,
    K: int,
) -> Dict[str, torch.Tensor]:
    """
    For theta/beta: block-wise average over active clients only.
    For log_psi: standard mean over all clients.
    """
    EPS = 1e-12
    M = len(client_deltas)

    out = {}
    for key in global_state.keys():
        updates = torch.stack([d[key] for d in client_deltas], dim=0)

        if key == "theta":
            w = active[:, None, :, None]
            num = (updates * w).sum(dim=0)
            den = w.sum(dim=0)
            agg = num / (den + EPS)
            agg = torch.where(den > 0, agg, torch.zeros_like(agg))
            out[key] = agg

        elif key == "beta":
            w = active[:, None, :, None]
            num = (updates * w).sum(dim=0)
            den = w.sum(dim=0)
            agg = num / (den + EPS)
            agg = torch.where(den > 0, agg, torch.zeros_like(agg))
            out[key] = agg

        else:
            out[key] = updates.mean(dim=0)

    return out


def run_simulation(data_mode, basis_mode):
    K = 5
    G = 1000
    N_CLIENTS = 8
    ROUNDS = 1000
    L1_LAMBDA = 2e-7

    FISHER_SPOT_SAMPLE = 100

    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    missing_celltypes_by_client = {
        2: [4],
        5: [1, 3],
        7: [0, 2, 4]
    }

    print("Loading data from combined CSVs...")

    DATA_DIR = "./sim_data/"
    Y_CSV_PATH = DATA_DIR + f"AllClients_Y_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{1000}_h_{data_mode}.csv"
    OMEGA_S_CSV_PATH = DATA_DIR + f"Input_Omega_S_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{1000}_h_{data_mode}.csv"

    clients_data, client_ids = load_clients_from_combined_csv(
        y_csv_path=Y_CSV_PATH,
        omega_s_csv_path=OMEGA_S_CSV_PATH,
        G=G,
        K=K,
        device="cpu",
        gene_prefix="Gene_",
        omega_prefix="Omega_Cell_",
        missing_celltypes_by_client=missing_celltypes_by_client,
    )

    full_s_coords = torch.cat([d["s"].cpu() for d in clients_data], dim=0)
    print(f"full_s_coords shape = {tuple(full_s_coords.shape)}")

    print(f">>> Start FedAdamW (G={G}, K={K}) | Clients={len(clients_data)}")
    basis_gen = BasisGenerator(mode=basis_mode, s_coords=full_s_coords)
    L_dim = basis_gen.L

    BETA_TRUE_CSV_PATH = DATA_DIR + f"GlobalTrue_Beta_G{G}_K{K}_L1_{L1_LAMBDA}_h_{data_mode}.csv"
    global_true_beta = load_global_true_beta(
        beta_csv_path=BETA_TRUE_CSV_PATH,
        G=G,
        K=K,
        L_dim=L_dim,
        device="cpu"
    )
    print(f"Loaded global_true_beta shape = {tuple(global_true_beta.shape)}  (L,K,G)")

    clients: List[FederatedClient] = []
    for idx, cid in enumerate(client_ids):
        data = clients_data[idx]
        miss = missing_celltypes_by_client.get(int(cid), [])

        if idx == 0:
            print("\n[Check Dimensions]")
            print(f"  - y: {tuple(data['y'].shape)}  (spots, genes)")
            print(f"  - omega: {tuple(data['omega'].shape)}  (spots, K)")
            print(f"  - s: {tuple(data['s'].shape)}  (spots, 2)\n")

        clients.append(FederatedClient(
            cid, data, K, G, basis_gen, L1_LAMBDA,
            device="cpu", missing_cell_types=miss
        ))

    if 2 in client_ids:
        c2_idx = client_ids.index(2)
        print("Sanity: client2 omega col4 max =",
              clients[c2_idx].data["omega"][:, 4].max().item())

    global_model = CTSVModel(K, G, basis_gen)
    global_state = global_model.state_dict()

    global_param_names = [name for name, _ in global_model.named_parameters()]
    print("Global param order:", global_param_names)

    print("Initializing theta from data mean...")
    all_means = []
    for client in clients:
        local_mean = torch.log(torch.mean(client.data['y'].float(), dim=0) + 1.0)
        all_means.append(local_mean)
    avg_log_mean = torch.stack(all_means).mean(dim=0)
    global_state['theta'] = avg_log_mean.unsqueeze(0).unsqueeze(0).repeat(1, K, 1)

    global_delta_G = [torch.zeros_like(dict(global_model.named_parameters())[name]) for name in global_param_names]

    loss_history = []
    grad_norm_history = []

    print("\nTraining (block-wise FedAvg with missing blocks)...")
    for r in range(ROUNDS):
        round_losses = []
        client_deltas = []

        for client in clients:
            d_state, loss = client.local_update(global_state, global_delta_G)
            round_losses.append(loss)
            client_deltas.append(d_state)

        avg_loss = float(np.mean(round_losses))
        loss_history.append(avg_loss)

        active = build_active_matrix(clients, K, device=global_state["theta"].device)

        global_delta_aggregated = aggregate_deltas_blockwise(
            client_deltas=client_deltas,
            global_state=global_state,
            active=active,
            K=K
        )

        total_norm_sq = 0.0
        for key in global_state:
            total_norm_sq += torch.sum(global_delta_aggregated[key] ** 2).item()
            global_state[key] += global_delta_aggregated[key]

        grad_norm = float(np.sqrt(total_norm_sq))
        grad_norm_history.append(grad_norm)

        global_delta_G = [global_delta_aggregated[name] for name in global_param_names]

        if r % 10 == 0:
            act_k4 = int(active[:, 4].sum().item())
            print(f"Round {r:3d} | Loss: {avg_loss:.4f} | Grad Norm: {grad_norm:.6f} | active(k=4)={act_k4}")

    print("\n>>> Training done. Compute Empirical Fisher diag (beta)...")

    global_fisher = np.zeros((L_dim, K, G), dtype=np.float64)
    for client in clients:
        f_local = client.compute_empirical_fisher_diag_beta(
            global_state,
            fisher_spot_sample=FISHER_SPOT_SAMPLE
        )
        global_fisher += f_local

    global_variance = 1.0 / (global_fisher + 1e-10)
    global_stderr = np.sqrt(global_variance)

    se_flat = global_stderr.reshape(-1)
    print("Std_Error quantiles:", np.quantile(se_flat, [0.5, 0.9, 0.99]))

    print("\nPlotting...")
    beta_pred = global_state['beta'].cpu().detach().numpy()
    beta_true = global_true_beta.cpu().detach().numpy()

    plot_genes = [130, 250, 400, 550, 660]
    fig, axes = plt.subplots(2, 6, figsize=(28, 8))

    axes[0, 0].plot(loss_history)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)

    axes[1, 0].plot(grad_norm_history)
    axes[1, 0].set_title('Grad Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    for i, g_idx in enumerate(plot_genes):
        col = i + 1
        title_str = f"Gene {g_idx}\n(block-wise; client2 no ct4)"

        sns.heatmap(beta_true[:, :, g_idx], ax=axes[0, col], cmap="RdBu_r",
                    center=0, vmin=-5, vmax=5, annot=True, fmt=".1f")
        axes[0, col].set_title(f"{title_str} [True]")
        axes[0, col].set_xlabel("Cell Types (0-4)")

        sns.heatmap(beta_pred[:, :, g_idx], ax=axes[1, col], cmap="RdBu_r",
                    center=0, vmin=-5, vmax=5, annot=True, fmt=".1f")
        axes[1, col].set_title(f"{title_str} [Learned]")
        axes[1, col].set_xlabel("Cell Types (0-4)")

    fig.suptitle(
        f"FedAdamW BLOCK-WISE (G={G}, K={K}) | L1={L1_LAMBDA} | FisherSample={FISHER_SPOT_SAMPLE}",
        fontsize=16
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    print("\nSaving CSV...")
    csv_rows = []
    for g in range(G):
        for k in range(K):
            for l in range(L_dim):
                feat_name = "h1" if l == 0 else "h2"
                val_true = beta_true[l, k, g]
                val_pred = beta_pred[l, k, g]
                val_var = global_variance[l, k, g]
                val_se = global_stderr[l, k, g]
                z_score = val_pred / (val_se + 1e-10)

                csv_rows.append({
                    "Gene_ID": g,
                    "Cell_Type": k,
                    "Feature": feat_name,
                    "True_Beta": float(val_true),
                    "Learned_Beta": float(val_pred),
                    "Variance": float(val_var),
                    "Std_Error": float(val_se),
                    "Z_Score": float(z_score),
                    "Diff": float(val_pred - val_true)
                })

    out_csv = (
        f"CTSV_Fed_BLOCKWISE_missingBlocks_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{ROUNDS}_"
        f"h_DataMode_{data_mode}_BasisMode_{basis_mode}_Inference_FIXED.csv"
    )
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Done! Saved: {out_csv}")


if __name__ == "__main__":
    experiments_list = [
        ("linear", "gau1"),
        ("linear", "gau2"),
        ("linear", "cos1"),
        ("linear", "cos2"),
        ("linear", "linear"),
        ("cos", "gau1"),
        ("cos", "gau2"),
        ("cos", "cos1"),
        ("cos", "cos2"),
        ("cos", "linear")
    ]

    print(f"总共计划运行 {len(experiments_list)} 个实验组合。\n")

    total_start = time.time()

    for idx, (d_mode, b_mode) in enumerate(experiments_list):
        print(f"--- 进度: {idx + 1}/{len(experiments_list)} ---")

        try:
            run_simulation(data_mode=d_mode, basis_mode=b_mode)

        except Exception as e:
            print(f"\n!!! 组合 [Data={d_mode}, Basis={b_mode}] 运行出错 !!!")
            print(f"错误信息: {e}")
            import traceback

            traceback.print_exc()
            print("正在跳过此组合，继续下一个...\n")
            continue

    total_end = time.time()
    print(f"\n所有任务全部完成！总耗时: {(total_end - total_start) / 60:.2f} 分钟。")