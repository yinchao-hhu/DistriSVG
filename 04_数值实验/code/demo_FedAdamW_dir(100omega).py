import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BasisGenerator:
    def __init__(self, mode='focal'):
        self.mode = mode.lower()
        self.L = 2

    def compute_h(self, s, device='cpu'):
        s = s.to(device)
        s_mean = torch.mean(s, dim=0, keepdim=True)
        s_std = torch.std(s, dim=0, keepdim=True) + 1e-8
        s_norm = (s - s_mean) / s_std

        s1 = s_norm[:, 0]
        s2 = s_norm[:, 1]

        if self.mode == 'linear':
            return s_norm
        elif self.mode == 'focal':
            sigma = 1.0
            h1 = torch.exp(-(s1 ** 2) / (2 * sigma ** 2))
            h2 = torch.exp(-(s2 ** 2) / (2 * sigma ** 2))
            return torch.stack([h1, h2], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def nb_loss(model, data_batch):
    y_true = data_batch['y']
    s = data_batch['s']
    omega = data_batch['omega']

    mu_pred = model(s, omega)
    psi = model.get_psi()

    probs = psi / (psi + mu_pred + 1e-8)
    dist = torch.distributions.NegativeBinomial(total_count=psi, probs=probs)
    nll = -dist.log_prob(y_true).mean()

    return nll


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

    def get_psi(self): return torch.exp(self.log_psi)


class FederatedClient:
    def __init__(self, client_id, data, n_cell_types, n_genes, basis_gen, l1_lambda, device='cpu'):
        self.id = client_id
        self.data = data
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
            loss = nb_loss(model, self.data)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                beta1, beta2 = 0.9, 0.99
                epsilon = 1e-8
                for i, p in enumerate(params):
                    grad = p.grad
                    if grad is None: continue
                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
                    m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                    v_hat = v[i] / (1 - beta2 ** (epoch + 1))
                    step = m_hat / (torch.sqrt(v_hat) + epsilon)

                    drift = 0
                    if global_delta_G is not None: drift = self.alpha * global_delta_G[i]
                    p_temp = p.data - current_lr * (step + drift)

                    if p.shape == model.beta.shape:
                        base_thresh = current_lr * self.l1_lambda
                        p.data = soft_threshold_adaptive(p_temp, base_thresh, v_hat, epsilon)
                    else:
                        p.data = p_temp
            current_lr *= 0.98

        delta = {}
        state = model.state_dict()
        for k in global_model_state: delta[k] = state[k] - global_model_state[k]
        return delta, loss.item()

    def compute_hessian_diag(self, global_model_state):
        model = CTSVModel(self.n_cell_types, self.n_genes, self.basis_gen).to(self.device)
        model.load_state_dict(global_model_state)
        model.eval()

        loss = nb_loss(model, self.data)
        N = self.data['y'].shape[0]
        total_loss = loss * N

        grads = torch.autograd.grad(total_loss, model.beta, create_graph=True)[0]

        hessian_diag = grads.pow(2)

        return hessian_diag.detach().cpu().numpy()


def run_simulation():
    BASIS_MODE = 'focal'
    K = 5
    G = 1000
    N_CLIENTS = 8
    ROUNDS = 1000
    L1_LAMBDA = 0.0000002

    print(f">>> 启动 FedAdamW (G={G}, K={K}) | Clients={N_CLIENTS}")

    basis_gen = BasisGenerator(mode=BASIS_MODE)
    L_dim = basis_gen.L

    print("生成 Theta...")
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

    print("生成 Beta...")
    global_true_beta = torch.zeros(L_dim, K, G)

    def get_random_signal(shape):
        mag = torch.rand(shape) * 3.0 + 2.0
        sign = torch.randint(0, 2, shape).float() * 2 - 1
        return mag * sign

    for k in range(K):
        start = (k + 1) * 100
        end = start + 200
        if end > G: end = G
        interval_len = end - start
        if interval_len > 0:
            global_true_beta[:, k, start:end] = get_random_signal((L_dim, interval_len))
            print(f"  - Cell {k} Active: Genes {start}-{end - 1}")

    clients = []
    chunk_size = G // N_CLIENTS
    chunk_size_2 = 50
    print(f"生成数据...")
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
            print(f"\n[Check Dimensions]")
            print(f"  - Client {i} Data 'y' Shape: {data['y'].shape}")
            print(f"    -> (Spots={data['y'].shape[0]}, Genes={data['y'].shape[1]})")
            print(f"  - Client {i} Data 'omega' Shape: {data['omega'].shape}")
            print(f"  - Client {i} Data 's' Shape: {data['s'].shape}\n")
        clients.append(FederatedClient(i, data, K, G, basis_gen, L1_LAMBDA))

    global_model = CTSVModel(K, G, basis_gen)
    global_state = global_model.state_dict()

    print("初始化 Theta...")
    all_means = []
    for client in clients:
        local_mean = torch.log(torch.mean(client.data['y'].float(), dim=0) + 1.0)
        all_means.append(local_mean)
    avg_log_mean = torch.stack(all_means).mean(dim=0)
    global_state['theta'] = avg_log_mean.unsqueeze(0).unsqueeze(0).repeat(1, K, 1)

    global_delta_G = [torch.zeros_like(p) for p in global_model.parameters()]
    loss_history = []
    grad_norm_history = []

    print("\n开始训练...")
    for r in range(ROUNDS):
        round_losses = []
        client_deltas = []
        for client in clients:
            d_theta, loss = client.local_update(global_state, global_delta_G)
            round_losses.append(loss)
            client_deltas.append(d_theta)

        avg_loss = np.mean(round_losses)
        loss_history.append(avg_loss)

        global_delta_aggregated = {}
        total_norm_sq = 0.0
        for key in global_state:
            updates = torch.stack([d[key] for d in client_deltas])
            agg = torch.mean(updates, dim=0)
            global_delta_aggregated[key] = agg
            total_norm_sq += torch.sum(agg ** 2).item()

        grad_norm = np.sqrt(total_norm_sq)
        grad_norm_history.append(grad_norm)

        for key in global_state:
            global_state[key] += global_delta_aggregated[key]
        global_delta_G = [global_delta_aggregated[name] for name, _ in global_model.named_parameters()]

        if r % 10 == 0:
            print(f"Round {r:3d} | Loss: {avg_loss:.4f} | Grad Norm: {grad_norm:.6f}")

    print("\n>>> 训练完成，开始 Federated Wald Inference (计算 Hessian)...")

    global_hessian = np.zeros((L_dim, K, G))

    for i, client in enumerate(clients):
        h_local = client.compute_hessian_diag(global_state)
        global_hessian += h_local

    global_variance = 1.0 / (global_hessian + 1e-10)
    global_stderr = np.sqrt(global_variance)

    print("Inference metrics computed.")

    print("\n>>> 绘图分析...")
    beta_pred = global_state['beta'].cpu().detach().numpy()
    beta_true = global_true_beta.cpu().detach().numpy()

    plot_genes = [30, 220, 400, 750, 900]

    fig, axes = plt.subplots(2, 6, figsize=(28, 8))

    axes[0, 0].plot(loss_history, 'b-')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    axes[1, 0].plot(grad_norm_history, 'r-')
    axes[1, 0].set_title('Grad Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    for i, g_idx in enumerate(plot_genes):
        col = i + 1
        missing_client = g_idx // chunk_size
        title_str = f"Gene {g_idx}\n(Miss in C{missing_client})"

        sns.heatmap(beta_true[:, :, g_idx], ax=axes[0, col], cmap="RdBu_r", center=0, vmin=-5, vmax=5, annot=True,
                    fmt=".1f")
        axes[0, col].set_title(f"{title_str} [True]")
        axes[0, col].set_xlabel("Cell Types (0-4)")

        sns.heatmap(beta_pred[:, :, g_idx], ax=axes[1, col], cmap="RdBu_r", center=0, vmin=-5, vmax=5, annot=True,
                    fmt=".1f")
        axes[1, col].set_title(f"{title_str} [Learned]")
        axes[1, col].set_xlabel("Cell Types (0-4)")

    fig.suptitle(f"FedAdamW (G={G}, K={K}) | L1={L1_LAMBDA}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    print("\n>>> 保存基因结果到 CSV...")
    csv_rows = []
    for g in range(G):
        for k in range(K):
            for l in range(L_dim):
                feat_name = "h1_X" if l == 0 else "h2_Y"

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

    pd.DataFrame(csv_rows).to_csv(f"Beta_G{G}_L1_{L1_LAMBDA}_Inference.csv", index=False)
    print("完成!")


    print("\n>>> 正在保存输入数据 (Omega & Coordinates) 到单独的 CSV...")

    input_data_rows = []

    for client in clients:
        omega_val = client.data['omega'].cpu().detach().numpy()
        s_val = client.data['s'].cpu().detach().numpy()

        num_spots = omega_val.shape[0]

        for i in range(num_spots):
            row_dict = {
                "Client_ID": client.id,
                "Spot_ID": i,
                "Coord_X": float(s_val[i, 0]),
                "Coord_Y": float(s_val[i, 1])
            }

            for k in range(K):
                row_dict[f"Omega_Cell_{k}"] = float(omega_val[i, k])


            input_data_rows.append(row_dict)

    df_input = pd.DataFrame(input_data_rows)
    filename_input = f"Input_Data_Omega_S_Clients{N_CLIENTS}_G{G}_L1_{L1_LAMBDA}.csv"
    df_input.to_csv(filename_input, index=False)

    print(f"✅ 输入数据已保存: {filename_input}")
    print(f"   包含 Spot 总数: {len(df_input)}")
    print(f"   前 5 行预览:\n{df_input.head()}")

if __name__ == "__main__":
    run_simulation()