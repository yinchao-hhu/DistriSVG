from pathlib import Path
import argparse
import math
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise SystemExit(
        "This script requires PyTorch. Run it in the Python environment used for the original "
        "CTSV demos, or install torch in a local environment before running."
    ) from exc


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
DATA_DIR = ROOT / "code" / "new_test1-5" / "sim_data"
OUT_DIR = ROOT / "overleaf" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


METHODS = [
    ("AP-FedAdam", {"adam": True, "prox": True, "drift": True, "federated": True}),
    ("No-prox", {"adam": True, "prox": False, "drift": True, "federated": True}),
    ("No-drift", {"adam": True, "prox": True, "drift": False, "federated": True}),
    ("FedAvg", {"adam": False, "prox": False, "drift": False, "federated": True}),
    ("Local-only", {"adam": True, "prox": True, "drift": False, "federated": False}),
]

METHOD_COLORS = {
    "AP-FedAdam": (31, 119, 180),
    "No-prox": (214, 39, 40),
    "No-drift": (44, 160, 44),
    "FedAvg": (148, 103, 189),
    "Local-only": (255, 127, 14),
}


class BasisGenerator:
    def __init__(self, mode="cos2", s_coords=None):
        self.mode = mode.lower()
        self.L = 2
        if s_coords is None:
            s_coords = torch.rand(1000, 2)
        self.mean = torch.mean(s_coords, dim=0, keepdim=True)
        self.std = torch.std(s_coords, dim=0, keepdim=True) + 1e-8
        s_norm = (s_coords - self.mean) / self.std
        abs_s = torch.abs(s_norm)
        self.psi_40 = torch.tensor(
            [torch.quantile(abs_s[:, 0], 0.4), torch.quantile(abs_s[:, 1], 0.4)]
        )
        self.psi_60 = torch.tensor(
            [torch.quantile(abs_s[:, 0], 0.6), torch.quantile(abs_s[:, 1], 0.6)]
        )
        s1 = s_norm[:, 0]
        s2 = s_norm[:, 1]
        h1_gau1 = torch.exp(-(s1**2) / (2 * self.psi_40[0] ** 2))
        h2_gau1 = torch.exp(-(s2**2) / (2 * self.psi_40[1] ** 2))
        self.gau1_mean = torch.stack([h1_gau1.mean(), h2_gau1.mean()])
        self.gau1_std = torch.stack([h1_gau1.std() + 1e-8, h2_gau1.std() + 1e-8])
        h1_gau2 = torch.exp(-(s1**2) / (2 * self.psi_60[0] ** 2))
        h2_gau2 = torch.exp(-(s2**2) / (2 * self.psi_60[1] ** 2))
        self.gau2_mean = torch.stack([h1_gau2.mean(), h2_gau2.mean()])
        self.gau2_std = torch.stack([h1_gau2.std() + 1e-8, h2_gau2.std() + 1e-8])

    def compute_h(self, s, device="cpu"):
        s = s.to(device)
        s_mean = torch.mean(s, dim=0, keepdim=True)
        s_std = torch.std(s, dim=0, keepdim=True) + 1e-8
        s_norm = (s - s_mean) / s_std
        s1 = s_norm[:, 0]
        s2 = s_norm[:, 1]
        psi40 = self.psi_40.to(device)
        psi60 = self.psi_60.to(device)
        if self.mode == "linear":
            return s_norm
        if self.mode == "gau1":
            h = torch.stack(
                [
                    torch.exp(-(s1**2) / (2 * psi40[0] ** 2)),
                    torch.exp(-(s2**2) / (2 * psi40[1] ** 2)),
                ],
                dim=1,
            )
            return (h - self.gau1_mean.to(device)) / self.gau1_std.to(device)
        if self.mode == "gau2":
            h = torch.stack(
                [
                    torch.exp(-(s1**2) / (2 * psi60[0] ** 2)),
                    torch.exp(-(s2**2) / (2 * psi60[1] ** 2)),
                ],
                dim=1,
            )
            return (h - self.gau2_mean.to(device)) / self.gau2_std.to(device)
        if self.mode == "cos1":
            return torch.stack([torch.cos(2 * np.pi * s1 / psi40[0]), torch.cos(2 * np.pi * s2 / psi40[1])], dim=1)
        if self.mode == "cos2":
            return torch.stack([torch.cos(2 * np.pi * s1 / psi60[0]), torch.cos(2 * np.pi * s2 / psi60[1])], dim=1)
        raise ValueError(f"Unknown basis mode: {self.mode}")


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
        spatial_part = torch.einsum("nl,lkg->nkg", h, self.beta)
        eta = self.theta + spatial_part
        log_mu = torch.sum(omega.unsqueeze(-1) * eta, dim=1)
        return torch.exp(log_mu)

    def get_psi(self):
        return torch.exp(self.log_psi)


def nb_log_prob_matrix(mu, psi, y):
    denom = psi + mu + 1e-8
    probs = torch.clamp(psi / denom, min=1e-6, max=1.0 - 1e-6)
    dist = torch.distributions.NegativeBinomial(total_count=psi, probs=probs)
    return dist.log_prob(y)


def nb_loss_mean(model, data_batch):
    mu = model(data_batch["s"], data_batch["omega"])
    logp = nb_log_prob_matrix(mu, model.get_psi(), data_batch["y"])
    return -logp.mean()


def soft_threshold_adaptive(w, base_lambda, v_hat, epsilon=1e-8):
    adaptive_threshold = base_lambda / (torch.sqrt(v_hat) + epsilon)
    return torch.sign(w) * torch.maximum(torch.abs(w) - adaptive_threshold, torch.zeros_like(w))


def clone_state(state):
    return {k: v.detach().clone() for k, v in state.items()}


def load_clients(data_mode, gene_limit, k, device):
    y_path = DATA_DIR / f"AllClients_Y_8_G_1000_L1_2e-07_Round1000_h_{data_mode}.csv"
    os_path = DATA_DIR / f"Input_Omega_S_8_G_1000_L1_2e-07_Round1000_h_{data_mode}.csv"
    df_y = pd.read_csv(y_path)
    df_o = pd.read_csv(os_path)
    gene_cols = [f"Gene_{i}" for i in range(gene_limit)]
    omega_cols = [f"Omega_Cell_{i}" for i in range(k)]
    df = df_y[["Client_ID", "Spot_ID"] + gene_cols].merge(
        df_o[["Client_ID", "Spot_ID", "Coord_X", "Coord_Y"] + omega_cols],
        on=["Client_ID", "Spot_ID"],
        how="inner",
        validate="one_to_one",
    )
    df = df.sort_values(["Client_ID", "Spot_ID"]).reset_index(drop=True)
    clients = []
    for cid in sorted(df["Client_ID"].unique().tolist()):
        sub = df[df["Client_ID"] == cid]
        omega = sub[omega_cols].to_numpy(dtype=np.float32)
        omega = np.clip(omega, 1e-8, None)
        omega = omega / omega.sum(axis=1, keepdims=True)
        clients.append(
            {
                "id": cid,
                "y": torch.tensor(sub[gene_cols].to_numpy(dtype=np.float32), device=device),
                "s": torch.tensor(sub[["Coord_X", "Coord_Y"]].to_numpy(dtype=np.float32), device=device),
                "omega": torch.tensor(omega, dtype=torch.float32, device=device),
            }
        )
    return clients


def load_true_beta(data_mode, gene_limit, k, l_dim, device):
    path = DATA_DIR / f"GlobalTrue_Beta_G1000_K5_L1_2e-07_h_{data_mode}.csv"
    df = pd.read_csv(path)
    df = df[df["Gene_ID"] < gene_limit]
    beta = np.zeros((l_dim, k, gene_limit), dtype=np.float32)
    for _, row in df.iterrows():
        l = 0 if row["Feature"] == "h1" else 1
        beta[l, int(row["Cell_Type"]), int(row["Gene_ID"])] = float(row["True_Beta"])
    return torch.tensor(beta, dtype=torch.float32, device=device)


def initialize_global_state(clients, k, g, basis_gen, device):
    model = CTSVModel(k, g, basis_gen).to(device)
    state = clone_state(model.state_dict())
    local_means = [torch.log(torch.mean(client["y"].float(), dim=0) + 1.0) for client in clients]
    avg_log_mean = torch.stack(local_means).mean(dim=0)
    state["theta"] = avg_log_mean.unsqueeze(0).unsqueeze(0).repeat(1, k, 1).detach().clone()
    return state


def local_update(state, data, k, g, basis_gen, config, l1_lambda, global_delta, local_steps, lr, drift_scale):
    model = CTSVModel(k, g, basis_gen)
    model.load_state_dict(state)
    params = list(model.parameters())
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) + 1e-8 for p in params]
    current_lr = lr
    last_loss = None
    beta1, beta2 = 0.9, 0.99
    eps = 1e-8
    for epoch in range(local_steps):
        loss = nb_loss_mean(model, data)
        last_loss = float(loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for i, p in enumerate(params):
                grad = p.grad
                if grad is None:
                    continue
                if config["adam"]:
                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    v[i] = beta2 * v[i] + (1 - beta2) * (grad**2)
                    m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                    v_hat = v[i] / (1 - beta2 ** (epoch + 1))
                    step = m_hat / (torch.sqrt(v_hat) + eps)
                else:
                    v_hat = torch.ones_like(p)
                    step = grad
                drift = 0.0
                if config["drift"] and global_delta is not None:
                    drift = drift_scale * global_delta[i]
                p_temp = p.data - current_lr * (step + drift)
                if config["prox"] and p.shape == model.beta.shape:
                    p.data = soft_threshold_adaptive(p_temp, current_lr * l1_lambda, v_hat, eps)
                else:
                    p.data = p_temp
        current_lr *= 0.98
    new_state = clone_state(model.state_dict())
    delta = {key: new_state[key] - state[key] for key in state}
    return new_state, delta, last_loss


def state_update_norm(delta):
    total = 0.0
    for value in delta.values():
        total += torch.sum(value**2).item()
    return math.sqrt(total)


def beta_metrics(beta_pred, beta_true):
    pred = beta_pred.detach().cpu().numpy().reshape(-1)
    true = beta_true.detach().cpu().numpy().reshape(-1)
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(true, pred)[0, 1]) if np.std(pred) > 0 and np.std(true) > 0 else float("nan")
    pred_zero_frac = float(np.mean(np.abs(pred) < 1e-8))
    pred_small_frac = float(np.mean(np.abs(pred) < 1e-4))
    true_zero_frac = float(np.mean(np.abs(true) < 1e-8))
    beta_l1_norm = float(np.sum(np.abs(pred)))
    return mae, rmse, corr, pred_zero_frac, pred_small_frac, true_zero_frac, beta_l1_norm


def run_method(method_name, config, clients, k, g, basis_gen, true_beta, args):
    start = time.time()
    history_rows = []
    init_state = initialize_global_state(clients, k, g, basis_gen, args.device)
    if config["federated"]:
        global_state = clone_state(init_state)
        param_names = list(CTSVModel(k, g, basis_gen).state_dict().keys())
        global_delta = [torch.zeros_like(v) for v in CTSVModel(k, g, basis_gen).parameters()]
        for round_idx in range(args.rounds):
            losses = []
            deltas = []
            for client in clients:
                _, delta, loss = local_update(
                    global_state, client, k, g, basis_gen, config, args.l1_lambda, global_delta,
                    args.local_steps, args.lr, args.drift_scale
                )
                losses.append(loss)
                deltas.append(delta)
            agg_delta = {}
            for key in global_state:
                agg_delta[key] = torch.stack([delta[key] for delta in deltas]).mean(dim=0)
                global_state[key] = global_state[key] + agg_delta[key]
            global_delta = [agg_delta[name] for name in param_names]
            history_rows.append(
                {
                    "method": method_name,
                    "round": round_idx,
                    "loss": float(np.mean(losses)),
                    "update_norm": state_update_norm(agg_delta),
                }
            )
            if round_idx % max(1, args.print_every) == 0:
                print(f"{method_name:10s} round {round_idx:4d} loss={np.mean(losses):.4f}")
        beta_pred = global_state["beta"]
        mae, rmse, corr, zero_frac, small_frac, true_zero_frac, beta_l1 = beta_metrics(beta_pred, true_beta)
    else:
        local_states = [clone_state(init_state) for _ in clients]
        for round_idx in range(args.rounds):
            losses = []
            norms = []
            for i, client in enumerate(clients):
                new_state, delta, loss = local_update(
                    local_states[i], client, k, g, basis_gen, config, args.l1_lambda, None,
                    args.local_steps, args.lr, args.drift_scale
                )
                local_states[i] = new_state
                losses.append(loss)
                norms.append(state_update_norm(delta))
            history_rows.append(
                {
                    "method": method_name,
                    "round": round_idx,
                    "loss": float(np.mean(losses)),
                    "update_norm": float(np.mean(norms)),
                }
            )
            if round_idx % max(1, args.print_every) == 0:
                print(f"{method_name:10s} round {round_idx:4d} loss={np.mean(losses):.4f}")
        maes, rmses, corrs = [], [], []
        for state in local_states:
            mae, rmse, corr, _, _, _, _ = beta_metrics(state["beta"], true_beta)
            maes.append(mae)
            rmses.append(rmse)
            corrs.append(corr)
        mae, rmse, corr = float(np.mean(maes)), float(np.mean(rmses)), float(np.nanmean(corrs))
        zero_frac = float("nan")
        small_frac = float("nan")
        true_zero_frac = float("nan")
        beta_l1 = float("nan")
    elapsed = time.time() - start
    summary = {
        "method": method_name,
        "l1_lambda": args.l1_lambda,
        "drift_scale": args.drift_scale,
        "data_mode": args.data_mode,
        "basis_mode": args.basis_mode,
        "rounds": args.rounds,
        "local_steps": args.local_steps,
        "gene_limit": g,
        "final_loss": history_rows[-1]["loss"],
        "final_update_norm": history_rows[-1]["update_norm"],
        "final_mae": mae,
        "final_rmse": rmse,
        "final_pearson": corr,
        "pred_zero_frac_1e8": zero_frac,
        "pred_small_frac_1e4": small_frac,
        "true_zero_frac_1e8": true_zero_frac,
        "beta_l1_norm": beta_l1,
        "elapsed_sec": elapsed,
    }
    return history_rows, summary


def font(size=24, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_line_plot(df, y_col, out_path, title, y_label, log_y=False):
    methods = df["method"].drop_duplicates().tolist()
    w, h = 1120, 720
    left, top, plot_w, plot_h = 120, 110, 780, 480
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(34, bold=True)
    label_font = font(23, bold=True)
    small_font = font(20)
    draw.text((40, 34), title, fill=(20, 20, 20), font=title_font)
    vals = df[y_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if log_y:
        vals = vals[vals > 0]
        ymin = float(np.floor(np.log10(vals.min())))
        ymax = float(np.ceil(np.log10(vals.max())))
    else:
        ymin = float(vals.min())
        ymax = float(vals.max())
        pad = 0.08 * (ymax - ymin + 1e-8)
        ymin = 0.0 if ymin >= 0 else ymin - pad
        ymax += pad
    max_round = int(df["round"].max())

    def px(r):
        return left + r / max(1, max_round) * plot_w

    def py(v):
        vv = math.log10(max(v, 1e-12)) if log_y else v
        return top + plot_h - (vv - ymin) / (ymax - ymin + 1e-12) * plot_h

    for tick in np.linspace(0, max_round, 6):
        x = px(tick)
        draw.line([(x, top), (x, top + plot_h)], fill=(225, 225, 225, 255))
        draw.text((x - 18, top + plot_h + 12), f"{int(tick)}", fill=(40, 40, 40), font=small_font)
    if log_y:
        y_ticks = np.arange(math.floor(ymin), math.ceil(ymax) + 1)
    else:
        y_ticks = np.linspace(ymin, ymax, 5)
    for tick in y_ticks:
        y = top + plot_h - (tick - ymin) / (ymax - ymin + 1e-12) * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225, 255))
        label = f"1e{int(tick)}" if log_y else f"{tick:.2f}"
        draw.text((left - 82, y - 10), label, fill=(40, 40, 40), font=small_font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(20, 20, 20), width=2)

    for method in methods:
        sub = df[df["method"] == method].sort_values("round")
        points = [(px(int(row["round"])), py(float(row[y_col]))) for _, row in sub.iterrows()]
        if len(points) > 1:
            draw.line(points, fill=METHOD_COLORS[method] + (255,), width=4)
    draw.text((left + plot_w / 2 - 45, top + plot_h + 55), "Round", fill=(20, 20, 20), font=label_font)
    draw.text((left, top - 34), y_label, fill=(20, 20, 20), font=label_font)
    lx, ly = left + plot_w + 45, top + 30
    for i, method in enumerate(methods):
        yy = ly + i * 42
        draw.line([(lx, yy + 10), (lx + 34, yy + 10)], fill=METHOD_COLORS[method] + (255,), width=5)
        draw.text((lx + 46, yy), method, fill=(20, 20, 20), font=small_font)
    img.save(out_path)


def draw_bar(summary_df, out_path):
    methods = summary_df["method"].drop_duplicates().tolist()
    w, h = 1050, 720
    left, top, plot_w, plot_h = 115, 110, 760, 430
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(34, bold=True)
    label_font = font(23, bold=True)
    small_font = font(19)
    value_font = font(19, bold=True)
    draw.text((40, 34), "C. Final Beta MAE by Ablation Method", fill=(20, 20, 20), font=title_font)
    vals = [float(summary_df.loc[summary_df["method"] == m, "final_mae"].iloc[0]) for m in methods]
    ymax = max(vals) * 1.25 + 1e-8
    for tick in np.linspace(0, ymax, 5):
        y = top + plot_h - tick / ymax * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225, 255))
        draw.text((left - 78, y - 10), f"{tick:.2f}", fill=(40, 40, 40), font=small_font)
    draw.line([(left, top), (left, top + plot_h)], fill=(20, 20, 20), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(20, 20, 20), width=2)
    gap = 24
    bar_w = (plot_w - gap * (len(methods) + 1)) / len(methods)
    for i, method in enumerate(methods):
        val = vals[i]
        x0 = left + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = top + plot_h - val / ymax * plot_h
        draw.rectangle([x0, y0, x1, top + plot_h], fill=METHOD_COLORS[method] + (220,))
        draw.text((x0 + 4, top + plot_h + 14), method.replace("-", "\n"), fill=(20, 20, 20), font=small_font)
        draw.text((x0 + 4, y0 - 28), f"{val:.3f}", fill=(20, 20, 20), font=value_font)
    draw.text((left, top - 34), "MAE", fill=(20, 20, 20), font=label_font)
    img.save(out_path)


def tagged_name(stem, tag, suffix):
    if not tag:
        return OUT_DIR / f"{stem}{suffix}"
    clean = tag.replace("/", "_").replace(" ", "_")
    return OUT_DIR / f"{stem}_{clean}{suffix}"


def make_figures(history_df, summary_df, output_tag):
    draw_line_plot(
        history_df,
        "loss",
        tagged_name("fig3a_loss_convergence", output_tag, ".png"),
        "A. Training Loss Across Ablation Methods",
        "Loss",
        log_y=False,
    )
    draw_line_plot(
        history_df,
        "update_norm",
        tagged_name("fig3b_gradnorm_convergence", output_tag, ".png"),
        "B. Update Norm Across Ablation Methods",
        "Norm",
        log_y=True,
    )
    draw_bar(summary_df, tagged_name("fig3c_ablation_mae_bar", output_tag, ".png"))


def selected_methods(methods_arg):
    wanted = [name.strip() for name in methods_arg.split(",") if name.strip()]
    method_map = {name: config for name, config in METHODS}
    unknown = sorted(set(wanted) - set(method_map))
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
    return [(name, method_map[name]) for name in wanted]


def parse_args():
    parser = argparse.ArgumentParser(description="Run AP-FedAdam ablation on cos data + cos2 basis.")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--gene-limit", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l1-lambda", type=float, default=2e-7)
    parser.add_argument("--drift-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--data-mode", default="cos")
    parser.add_argument("--basis-mode", default="cos2")
    parser.add_argument(
        "--methods",
        default=",".join(name for name, _ in METHODS),
        help="Comma-separated subset of methods to run.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for CSV and figure outputs, used for stress-test sweeps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    k = 5
    g = args.gene_limit
    clients = load_clients(args.data_mode, g, k, args.device)
    full_s = torch.cat([client["s"].cpu() for client in clients], dim=0)
    basis_gen = BasisGenerator(mode=args.basis_mode, s_coords=full_s)
    true_beta = load_true_beta(args.data_mode, g, k, basis_gen.L, args.device)

    all_history = []
    summaries = []
    methods = selected_methods(args.methods)
    print(
        f"Running ablation: data={args.data_mode}, basis={args.basis_mode}, "
        f"genes={g}, rounds={args.rounds}, local_steps={args.local_steps}, "
        f"lambda={args.l1_lambda:g}, methods={','.join(name for name, _ in methods)}"
    )
    for method_name, config in methods:
        history, summary = run_method(method_name, config, clients, k, g, basis_gen, true_beta, args)
        all_history.extend(history)
        summaries.append(summary)
        print(
            f"{method_name:10s} final MAE={summary['final_mae']:.4f} "
            f"loss={summary['final_loss']:.4f} elapsed={summary['elapsed_sec']:.1f}s"
        )

    history_df = pd.DataFrame(all_history)
    summary_df = pd.DataFrame(summaries)
    history_path = tagged_name("fig3_ablation_history", args.output_tag, ".csv")
    summary_path = tagged_name("fig3_ablation_summary", args.output_tag, ".csv")
    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    make_figures(history_df, summary_df, args.output_tag)
    print(f"Wrote {history_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tagged_name('fig3a_loss_convergence', args.output_tag, '.png')}")
    print(f"Wrote {tagged_name('fig3b_gradnorm_convergence', args.output_tag, '.png')}")
    print(f"Wrote {tagged_name('fig3c_ablation_mae_bar', args.output_tag, '.png')}")


if __name__ == "__main__":
    main()
