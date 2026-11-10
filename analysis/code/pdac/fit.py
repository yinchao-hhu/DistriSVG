
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Optional, Dict


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
    Keeping your original training behavior.
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
    Critical: SUM over genes, not mean. This fixes the missing factor G issue.
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
    def __init__(self, client_name, client_id, data, n_cell_types, n_genes, basis_gen,
                 l1_lambda, device='cpu'):
        self.cid = client_name       
        self.id = client_id
        self.data = {k: v.to(device) for k, v in data.items()}
        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.basis_gen = basis_gen
        self.device = device

        self.E = 10
        self.lr = 0.05
        self.l1_lambda = l1_lambda
        self.alpha = 0.1

    def local_update(self, global_model_state, global_delta_G):
        model = CTSVModel(self.n_cell_types, self.n_genes, self.basis_gen).to(self.device)
        model.load_state_dict(global_model_state)
        params = list(model.parameters())

        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) + 1e-8 for p in params]
        current_lr = self.lr

        model.train()
        for epoch in range(self.E):
            loss = nb_loss_mean(model, self.data)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                beta1, beta2 = 0.9, 0.99
                eps = 1e-8
                for i, p in enumerate(params):
                    grad = p.grad
                    if grad is None:
                        continue

                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
                    m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                    v_hat = v[i] / (1 - beta2 ** (epoch + 1))

                    step = m_hat / (torch.sqrt(v_hat) + eps)

                    drift = 0.0
                    if global_delta_G is not None:
                        drift = self.alpha * global_delta_G[i]

                    p_temp = p.data - current_lr * (step + drift)

                    if p.shape == model.beta.shape:
                        base_thresh = current_lr * self.l1_lambda
                        p.data = soft_threshold_adaptive(p_temp, base_thresh, v_hat, eps)
                    else:
                        p.data = p_temp

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
        fisher_spot_sample:
            None  -> use all spots (accurate, slower)
            int   -> uniformly sample that many spots and rescale (faster)
        """
        model = CTSVModel(self.n_cell_types, self.n_genes, self.basis_gen).to(self.device)
        model.load_state_dict(global_model_state)
        model.eval()

        nll_per_spot = nb_nll_per_spot(model, self.data)
        N = nll_per_spot.shape[0]

        if fisher_spot_sample is None or fisher_spot_sample >= N:
            idx = torch.arange(N, device=self.device)
            scale = 1.0
        else:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed + self.id)
            perm = torch.randperm(N, generator=g, device=self.device)
            idx = perm[:fisher_spot_sample]
            scale = float(N) / float(fisher_spot_sample)

        fisher_diag = torch.zeros_like(model.beta)

        for j, i in enumerate(idx):
            gi = torch.autograd.grad(nll_per_spot[i], model.beta, retain_graph=True)[0]
            fisher_diag += gi.pow(2)

        fisher_diag = fisher_diag * scale
        return fisher_diag.detach().cpu().numpy()


def _read_table_auto(path: str) -> pd.DataFrame:
    """
    Robust reader: tries comma first, then tab if needed.
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep="\t")
        return df
    except Exception:
        return pd.read_csv(path, sep="\t")


def load_clients_from_separate_csvs(
    y_csv_paths: List[str],
    omega_with_xy_paths: List[str],
    G: Optional[int] = None,
    K: int = 15,
    device: str = "cpu",
    spot_col_y: str = "Spot_ID",
    spot_col_omega: str = "spot",
    x_col: str = "x",
    y_col: str = "y",
    celltype_cols: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """
    Read per-slice files:
      - Y file: columns = [Spot_ID, <gene1>, <gene2>, ..., <geneM>]  (M=19736)
      - Omega file: columns = [spot, x, y, <15 celltype cols>]

    Return:
      clients_data: list of dicts {'y','omega','s'} as torch tensors (per slice = one client)
      client_ids:   list of string ids derived from filenames (e.g., 'PDAC_A_ST1', ...)

    Robustness:
      - merges on Spot_ID/spot to guarantee alignment (validate one_to_one)
      - supports comma- or tab-delimited inputs (auto)
      - supports G=None (use all genes) or G=int (take first G genes by column order)
    """
    if len(y_csv_paths) != len(omega_with_xy_paths):
        raise ValueError(f"y_csv_paths and omega_with_xy_paths must have same length. "
                         f"Got {len(y_csv_paths)} vs {len(omega_with_xy_paths)}")

    if celltype_cols is None:
        raise ValueError(
            "celltype_cols is required. You must provide the 15 cell type names in order "
            "(e.g., ['Acinar','Cancer.clone.A',...,'Tuft'])."
        )

    if len(celltype_cols) != K:
        raise ValueError(f"len(celltype_cols)={len(celltype_cols)} must equal K={K}")

    clients_data: List[Dict[str, torch.Tensor]] = []
    client_ids: List[str] = []

    for y_path, o_path in zip(y_csv_paths, omega_with_xy_paths):
        df_y = _read_table_auto(y_path)
        df_o = _read_table_auto(o_path)

        if spot_col_y not in df_y.columns:
            raise ValueError(f"Y file {y_path} missing column: {spot_col_y}")
        for c in [spot_col_omega, x_col, y_col]:
            if c not in df_o.columns:
                raise ValueError(f"Omega file {o_path} missing column: {c}")

        gene_cols_all = [c for c in df_y.columns if c != spot_col_y]
        if len(gene_cols_all) == 0:
            raise ValueError(f"Y file {y_path} has no gene columns after removing {spot_col_y}")

        if G is None:
            gene_cols = gene_cols_all
        else:
            if G > len(gene_cols_all):
                raise ValueError(f"G={G} but only {len(gene_cols_all)} gene columns found in {y_path}")
            gene_cols = gene_cols_all[:G]

        if all(c in df_o.columns for c in celltype_cols):
            omega_cols = celltype_cols
        else:
            cand = [c for c in df_o.columns if c not in [spot_col_omega, x_col, y_col]]
            if len(cand) != K:
                raise ValueError(
                    f"Cannot determine omega columns in {o_path}: "
                    f"expected K={K} omega cols, but found {len(cand)} "
                    f"(excluding {spot_col_omega},{x_col},{y_col}).\n"
                    f"Tip: check celltype column names or pass celltype_cols explicitly."
                )
            omega_cols = cand

        if len(omega_cols) != K:
            raise ValueError(f"Omega columns length {len(omega_cols)} != K={K} for {o_path}")

        df = df_y[[spot_col_y] + gene_cols].merge(
            df_o[[spot_col_omega, x_col, y_col] + omega_cols],
            left_on=spot_col_y,
            right_on=spot_col_omega,
            how="inner",
            validate="one_to_one",
        )

        if df.shape[0] != df_y.shape[0] or df.shape[0] != df_o.shape[0]:
            y_keys = set(df_y[spot_col_y].astype(str).tolist())
            o_keys = set(df_o[spot_col_omega].astype(str).tolist())
            missing_in_o = list(y_keys - o_keys)[:8]
            missing_in_y = list(o_keys - y_keys)[:8]
            raise ValueError(
                f"Merge row count mismatch for slice:\n"
                f"  y={y_path} rows={df_y.shape[0]}\n"
                f"  o={o_path} rows={df_o.shape[0]}\n"
                f"  merged rows={df.shape[0]}\n"
                f"Examples missing in omega: {missing_in_o}\n"
                f"Examples missing in y: {missing_in_y}"
            )

        df = df.sort_values(by=[spot_col_y]).reset_index(drop=True)

        Y = df[gene_cols].to_numpy(dtype=np.float32)
        S = df[[x_col, y_col]].to_numpy(dtype=np.float32)
        Omega = df[omega_cols].to_numpy(dtype=np.float32)


        base = os.path.basename(y_path)
        cid = base.replace("_processed_data", "").replace(".csv", "")
        client_ids.append(cid)

        clients_data.append({
            "y": torch.tensor(Y, dtype=torch.float32, device=device),
            "s": torch.tensor(S, dtype=torch.float32, device=device),
            "omega": torch.tensor(Omega, dtype=torch.float32, device=device),
        })

    return clients_data, client_ids

















def run_method():
    BASIS_MODE = 'gau1'
    K = 15
    G = 4740
    N_CLIENTS = 6
    ROUNDS = 1000
    L1_LAMBDA = 2e-7

    chunk_size = G // N_CLIENTS

    FISHER_SPOT_SAMPLE = 100

    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    print("Loading data from separate PDAC slice CSVs...")

    FilePath = "./"
    DATA_DIR_Y = FilePath + "preprocessed_data"
    DATA_DIR_O = FilePath + "spotlight_out"

    y_files = [
        "PDAC_A_ST1_processed_data_mincell5.csv",
        "PDAC_A_ST2_processed_data_mincell5.csv",
        "PDAC_A_ST3_processed_data_mincell5.csv",
        "PDAC_B_ST1_processed_data_mincell5.csv",
        "PDAC_B_ST2_processed_data_mincell5.csv",
        "PDAC_B_ST3_processed_data_mincell5.csv",
    ]

    omega_files = [
        "PDAC_A_ST1_SPOTlight_W_omega_with_xy.csv",
        "PDAC_A_ST2_SPOTlight_W_omega_with_xy.csv",
        "PDAC_A_ST3_SPOTlight_W_omega_with_xy.csv",
        "PDAC_B_ST1_SPOTlight_W_omega_with_xy.csv",
        "PDAC_B_ST2_SPOTlight_W_omega_with_xy.csv",
        "PDAC_B_ST3_SPOTlight_W_omega_with_xy.csv",
    ]

    y_paths = [os.path.join(DATA_DIR_Y, f) for f in y_files]
    omega_paths = [os.path.join(DATA_DIR_O, f) for f in omega_files]

    G_use = None
    CELLTYPE_COLS_15 = [
        "Acinar", "Cancer.clone.A", "Cancer.clone.B", "Ductal", "Endocrine",
        "Endothelial", "Fibroblasts", "Macrophages", "Mast", "mDCs",
        "Monocytes", "pDCs", "RBCs", "T", "Tuft"
    ]

    clients_data, client_ids = load_clients_from_separate_csvs(
        y_csv_paths=y_paths,
        omega_with_xy_paths=omega_paths,
        G=G_use,
        K=K,
        device="cpu",
        spot_col_y="Spot_ID",
        spot_col_omega="spot",
        x_col="x",
        y_col="y",
        celltype_cols=CELLTYPE_COLS_15  
    )

    print(f"Loaded {len(clients_data)} clients: {client_ids}")
    print(f"Example shapes: y={tuple(clients_data[0]['y'].shape)}, "
        f"s={tuple(clients_data[0]['s'].shape)}, omega={tuple(clients_data[0]['omega'].shape)}")

    G = clients_data[0]["y"].shape[1]
    print(f"Using G={G}, K={K}")

    full_s_coords = torch.cat([d["s"].cpu() for d in clients_data], dim=0)
    print(f"full_s_coords shape = {tuple(full_s_coords.shape)}")

    print(f">>> Start FedAdamW (G={G}, K={K}) | Clients={len(clients_data)}")
    basis_gen = BasisGenerator(mode=BASIS_MODE, s_coords=full_s_coords)
    L_dim = basis_gen.L




    clients = []
    for idx, cid in enumerate(client_ids):
        data = clients_data[idx]
        print(idx)
        print(cid)

        if idx == 0:
            print("\n[Check Dimensions]")
            print(f"  - y: {tuple(data['y'].shape)}  (spots, genes)")
            print(f"  - omega: {tuple(data['omega'].shape)}  (spots, K)")
            print(f"  - s: {tuple(data['s'].shape)}  (spots, 2)\n")

        clients.append(FederatedClient(cid, idx, data, K, G, basis_gen, L1_LAMBDA))


    global_model = CTSVModel(K, G, basis_gen)
    global_state = global_model.state_dict()

    print("Initializing theta from data mean...")
    all_means = []
    for client in clients:
        local_mean = torch.log(torch.mean(client.data['y'].float(), dim=0) + 1.0)
        all_means.append(local_mean)
    avg_log_mean = torch.stack(all_means).mean(dim=0)
    global_state['theta'] = avg_log_mean.unsqueeze(0).unsqueeze(0).repeat(1, K, 1)

    global_delta_G = [torch.zeros_like(p) for p in global_model.parameters()]
    loss_history = []
    grad_norm_history = []

    print("\nTraining...")
    for r in range(ROUNDS):
        round_losses = []
        client_deltas = []

        for client in clients:
            d_state, loss = client.local_update(global_state, global_delta_G)
            round_losses.append(loss)
            client_deltas.append(d_state)

        avg_loss = float(np.mean(round_losses))
        loss_history.append(avg_loss)

        global_delta_aggregated = {}
        total_norm_sq = 0.0
        for key in global_state:
            updates = torch.stack([d[key] for d in client_deltas])
            agg = torch.mean(updates, dim=0)
            global_delta_aggregated[key] = agg
            total_norm_sq += torch.sum(agg ** 2).item()

        grad_norm = float(np.sqrt(total_norm_sq))
        grad_norm_history.append(grad_norm)

        for key in global_state:
            global_state[key] += global_delta_aggregated[key]

        global_delta_G = [global_delta_aggregated[name] for name, _ in global_model.named_parameters()]

        if r % 10 == 0:
            print(f"Round {r:3d} | Loss: {avg_loss:.4f} | Grad Norm: {grad_norm:.6f}")

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

    beta_pred = global_state['beta'].cpu().detach().numpy()








    print("\nSaving CSV...")
    csv_rows = []

    for g in range(G):
        for k in range(K):
            for l in range(L_dim):
                feat_name = "h1" if l == 0 else "h2"

                val_pred = beta_pred[l, k, g]
                val_var = global_variance[l, k, g]
                val_se = global_stderr[l, k, g]
                z_score = val_pred / (val_se + 1e-10)

                csv_rows.append({
                    "Gene_ID": g,
                    "Cell_Type": k,
                    "Feature": feat_name,
                    "Learned_Beta": float(val_pred),
                    "Variance": float(val_var),
                    "Std_Error": float(val_se),
                    "Z_Score": float(z_score)
                })

    out_csv = f"PDAC_CTSV_Fed_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{ROUNDS}_h_BasisMode_{BASIS_MODE}_Inference_FIXED.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Done! Saved: {out_csv}")


if __name__ == "__main__":
    run_method()
