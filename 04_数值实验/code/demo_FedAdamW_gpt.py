
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

        print(f"Basis Init [{self.mode}]: Psi_40={self.psi_40.numpy()}, Psi_60={self.psi_60.numpy()}")

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
            return torch.stack([h1, h2], dim=1)

        elif self.mode == 'gau2':
            scale = 2.0
            sigma1 = self.psi_60[0].to(device) * scale
            sigma2 = self.psi_60[1].to(device) * scale
            h1 = torch.exp(-(s1 ** 2) / (2 * sigma1 ** 2))
            h2 = torch.exp(-(s2 ** 2) / (2 * sigma2 ** 2))
            return torch.stack([h1, h2], dim=1)

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


def generate_synthetic_data(n_spots, n_cell_types, n_genes, basis_gen,
                            global_true_beta, global_true_theta,
                            mask_genes_indices=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    s = torch.rand(n_spots, 2)
    concentration = torch.full((n_cell_types,), 0.1)
    omega = torch.distributions.Dirichlet(concentration).sample((n_spots,))

    h = basis_gen.compute_h(s)

    effective_beta = global_true_beta.clone()
    if mask_genes_indices is not None:
        effective_beta[:, :, mask_genes_indices] = 0.0

    spatial_effect = torch.einsum('nl,lkg->nkg', h, effective_beta)
    eta = global_true_theta + spatial_effect

    log_mu = torch.sum(omega.unsqueeze(-1) * eta, dim=1)
    mu = torch.exp(log_mu)

    psi = 10.0
    p = psi / (psi + mu + 1e-8)
    y = torch.distributions.NegativeBinomial(total_count=psi, probs=p).sample()

    return {'y': y, 's': s, 'omega': omega}


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


def run_simulation():
    BASIS_MODE = 'gau2'
    K = 5
    G = 1000
    N_CLIENTS = 8
    ROUNDS = 800
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
            mask_genes_indices=dropout_indices,
            seed=42 + i
        )

        if i == 0:
            print("\n[Check Dimensions]")
            print(f"  - y: {data['y'].shape}  (spots, genes)")
            print(f"  - omega: {data['omega'].shape}  (spots, K)")
            print(f"  - s: {data['s'].shape}  (spots, 2)\n")

        clients.append(FederatedClient(i, data, K, G, basis_gen, L1_LAMBDA))

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
        missing_client = g_idx // chunk_size
        title_str = f"Gene {g_idx}\n(Miss in C{missing_client})"

        sns.heatmap(beta_true[:, :, g_idx], ax=axes[0, col], cmap="RdBu_r",
                    center=0, vmin=-5, vmax=5, annot=True, fmt=".1f")
        axes[0, col].set_title(f"{title_str} [True]")
        axes[0, col].set_xlabel("Cell Types (0-4)")

        sns.heatmap(beta_pred[:, :, g_idx], ax=axes[1, col], cmap="RdBu_r",
                    center=0, vmin=-5, vmax=5, annot=True, fmt=".1f")
        axes[1, col].set_title(f"{title_str} [Learned]")
        axes[1, col].set_xlabel("Cell Types (0-4)")

    fig.suptitle(f"FedAdamW (G={G}, K={K}) | L1={L1_LAMBDA} | FisherSample={FISHER_SPOT_SAMPLE}",
                 fontsize=16)
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

    out_csv = f"CTSV_Fed_{N_CLIENTS}_G_{G}_L1_{L1_LAMBDA}_Round{ROUNDS}_h_{BASIS_MODE}_Inference_FIXED.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Done! Saved: {out_csv}")

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


if __name__ == "__main__":
    run_simulation()
