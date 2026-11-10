
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

        elif self.mode == 'gau':
            h1 = torch.exp(-(s1 ** 2) / 2)
            h2 = torch.exp(-(s2 ** 2) / 2)
            h  = torch.stack([h1, h2], dim=1)
            h = (h - self.gau1_mean.to(device)) / self.gau1_std.to(device)
            return h

        elif self.mode == 'cos':
            h1 = torch.cos(2 * np.pi * s1)
            h2 = torch.cos(2 * np.pi * s2)
            return torch.stack([h1, h2], dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")












def generate_synthetic_data(n_spots, n_cell_types, n_genes, basis_gen,
                            global_true_beta, global_true_theta, alpha0=100,
                            mask_genes_indices=None, seed=42,
                            pi0=0.4, psi_val=10.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    s = torch.rand(n_spots, 2)

    concentration = torch.full((n_cell_types,), 0.1)
    omega_base = torch.distributions.Dirichlet(concentration).sample((n_spots,))

    eps = 1e-8
    omega_tmp = torch.clamp(omega_base, min=eps)
    omega_tmp = omega_tmp / omega_tmp.sum(dim=1, keepdim=True)
    conc = alpha0 * omega_tmp
    omega = torch.distributions.Dirichlet(conc).sample()

    h = basis_gen.compute_h(s)

    effective_beta = global_true_beta.clone()
    if mask_genes_indices is not None:
        effective_beta[:, :, mask_genes_indices] = 0.0

    spatial_effect = torch.einsum('nl,lkg->nkg', h, effective_beta)
    eta = global_true_theta + spatial_effect

    log_mu = torch.sum(omega_base.unsqueeze(-1) * eta, dim=1)
    mu = torch.exp(log_mu)

    psi = torch.tensor(psi_val, dtype=mu.dtype, device=mu.device)
    probs = psi / (psi + mu + 1e-8)
    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    nb = torch.distributions.NegativeBinomial(total_count=psi, probs=probs)
    y_nb = nb.sample()

    zero_mask = torch.bernoulli(torch.full_like(y_nb, pi0)).bool()
    y = torch.where(zero_mask, torch.zeros_like(y_nb), y_nb)

    return {'y': y, 's': s, 'omega_base': omega_base, 'omega': omega}


class FederatedClient:
    def __init__(self, client_id, data, n_cell_types, n_genes, basis_gen,
                 l1_lambda, device='cpu'):
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


def run_generate():
    BASIS_MODE = 'linear'
    K = 5
    G = 1000
    N_CLIENTS = 8
    ROUNDS = 1000
    L1_LAMBDA = 2e-7

    FISHER_SPOT_SAMPLE = 100

    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    print("Preloading coords for quantiles...")
    all_s_coords = []
    for i in range(N_CLIENTS):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        s_temp = torch.rand(500, 2)
        all_s_coords.append(s_temp)
    full_s_coords = torch.cat(all_s_coords, dim=0)

    print(f">>> Start FedAdamW (G={G}, K={K}) | Clients={N_CLIENTS}")
    basis_gen = BasisGenerator(mode=BASIS_MODE, s_coords=full_s_coords)
    L_dim = basis_gen.L

    print("Generating Theta...")
    scrna_base = torch.normal(mean=2.0, std=0.2, size=(G,))
    thetas_list = []
    DE_num = int(G * 0.3)
    all_genes = np.arange(G)

    for k in range(K):
        t = scrna_base.clone()
        if k > 0:
            de_idx = np.random.choice(all_genes, DE_num, replace=False)
            mean_val = 3.0 + (k % 2) * 1.0 + (k // 2) * 0.5
            t[de_idx] = torch.normal(mean=mean_val, std=0.2, size=(DE_num,))
        thetas_list.append(t)

    global_true_theta = torch.stack(thetas_list, dim=0).unsqueeze(0)

    print("Generating Beta...")
    global_true_beta = torch.zeros(L_dim, K, G)

    def get_random_signal(shape):
        mag = torch.rand(shape) * 3.0 + 2.0
        sign = torch.randint(0, 2, shape).float() * 2 - 1
        return mag * sign

    for k in range(K):
        start = (k + 1) * 100
        end = min(start + 200, G)
        if end > start:
            global_true_beta[:, k, start:end] = get_random_signal((L_dim, end - start))
            print(f"  - Cell {k} Active: Genes {start}-{end - 1}")

    print("Generating data...")
    clients = []
    chunk_size = G // N_CLIENTS
    chunk_size_2 = 50

    for i in range(N_CLIENTS):
        start = i * chunk_size
        end = start + chunk_size_2
        dropout_indices = list(range(start, end))

        data = generate_synthetic_data(
            n_spots=500, n_cell_types=K, n_genes=G,
            basis_gen=basis_gen,
            global_true_beta=global_true_beta,
            global_true_theta=global_true_theta,
            alpha0=100,
            mask_genes_indices=dropout_indices,
            seed=42 + i,
            pi0=0.4, psi_val=10.0
        )

        if i == 0:
            print("\n[Check Dimensions]")
            print(f"  - y: {data['y'].shape}  (spots, genes)")
            print(f"  - omega: {data['omega'].shape}  (spots, K)")
            print(f"  - s: {data['s'].shape}  (spots, 2)\n")

        clients.append(FederatedClient(i, data, K, G, basis_gen, L1_LAMBDA))


    print("\nSaving Input Data (Omega & Coordinates)...")
    input_rows = []
    for client in clients:
        omega_val = client.data['omega'].detach().cpu().numpy()
        s_val = client.data['s'].detach().cpu().numpy()
        num_spots = omega_val.shape[0]

        for i in range(num_spots):
            row = {
                "Client_ID": client.id,
                "Spot_ID": i,
                "Coord_X": float(s_val[i, 0]),
                "Coord_Y": float(s_val[i, 1])
            }
            for k in range(K):
                row[f"Omega_Cell_{k}"] = float(omega_val[i, k])
            input_rows.append(row)

    df_input = pd.DataFrame(input_rows)
    out_input = f"Input_Omega_S_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{ROUNDS}_h_{BASIS_MODE}.csv"
    df_input.to_csv(out_input, index=False)
    print(f"Saved: {out_input} | rows={len(df_input)}")

    print("\nSaving all clients Y...")
    all_y_dfs = []
    for client in clients:
        y_np = client.data['y'].detach().cpu().numpy()
        df_temp = pd.DataFrame(y_np, columns=[f"Gene_{gg}" for gg in range(G)])
        df_temp.insert(0, "Spot_ID", range(len(df_temp)))
        df_temp.insert(0, "Client_ID", client.id)
        all_y_dfs.append(df_temp)

    df_final_y = pd.concat(all_y_dfs, ignore_index=True)
    out_y = f"AllClients_Y_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{ROUNDS}_h_{BASIS_MODE}.csv"
    df_final_y.to_csv(out_y, index=False)
    print(f"Saved: {out_y} | shape={df_final_y.shape}")

    print("\nSaving Global TRUE Beta...")
    beta_true_np = global_true_beta.detach().cpu().numpy()

    beta_rows = []
    for g in range(G):
        for k in range(K):
            for l in range(L_dim):
                feat_name = "h1" if l == 0 else "h2"
                beta_rows.append({
                    "Gene_ID": g,
                    "Cell_Type": k,
                    "Feature": feat_name,
                    "True_Beta": float(beta_true_np[l, k, g])
                })

    df_beta_true = pd.DataFrame(beta_rows)

    out_beta_true = f"GlobalTrue_Beta_G{G}_K{K}_L1_{L1_LAMBDA}_h_{BASIS_MODE}_pi0_0.4_psi_10.csv"
    df_beta_true.to_csv(out_beta_true, index=False)

    print(f"Saved: {out_beta_true} | rows={len(df_beta_true)}")


if __name__ == "__main__":
    run_generate()
