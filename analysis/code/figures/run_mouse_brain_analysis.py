#!/usr/bin/env python3
"""AP-FedAdam CTSV analysis on mouse brain Visium data (3 sections as federated clients)."""

import math, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import squidpy as sq
from scipy import sparse
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
DATA_DIR = ROOT / "code" / "real_data_2" / "mouse_brain"
OUT_DIR = ROOT / "overleaf" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELLTYPE_MARKERS = {
    "Neurons": ["Snap25", "Syt1", "Rbfox3"],
    "Astrocytes": ["Gfap", "Aqp4"],
    "Oligodendrocytes": ["Mbp", "Mog", "Plp1"],
    "Microglia": ["Cx3cr1", "Aif1"],
    "Endothelial": ["Pecam1", "Cldn5"],
}

SECTIONS = [
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
]


def compute_celltype_proportions(adata, celltype_markers):
    """Compute marker-gene-based cell-type proportion scores per spot."""
    ct_names = list(celltype_markers.keys())
    scores = np.zeros((adata.shape[0], len(ct_names)))
    X = adata.X
    var_names = list(adata.var_names)
    for i, (ct, markers) in enumerate(celltype_markers.items()):
        present = [m for m in markers if m in adata.var_names]
        if not present:
            continue
        idx = [var_names.index(m) for m in present]
        vals = X[:, idx].astype(np.float64)
        if sparse.issparse(vals):
            vals = vals.copy()
            vals.data = np.log1p(vals.data)
            scores[:, i] = np.asarray(vals.mean(axis=1)).ravel()
        else:
            scores[:, i] = np.mean(np.log1p(vals), axis=1)
    scores = np.clip(scores, 0, None)
    row_sums = scores.sum(axis=1, keepdims=True) + 1e-8
    omega = scores / row_sums
    return dict(zip(ct_names, omega.T))


class BasisGenerator:
    def __init__(self, s_coords):
        self.L = 2
        self.mean = torch.mean(s_coords, dim=0, keepdim=True)
        self.std = torch.std(s_coords, dim=0, keepdim=True) + 1e-8

    def compute_h(self, s):
        s_norm = (s - self.mean) / self.std
        s1, s2 = s_norm[:, 0], s_norm[:, 1]
        return torch.stack([
            torch.cos(2 * math.pi * s1 / 0.6),
            torch.cos(2 * math.pi * s2 / 0.6),
        ], dim=1)


class CTSVModel(nn.Module):
    def __init__(self, n_cell_types, n_genes, basis_gen):
        super().__init__()
        self.K, self.G, self.basis_gen = n_cell_types, n_genes, basis_gen
        self.L = basis_gen.L
        self.theta = nn.Parameter(torch.zeros(1, self.K, self.G))
        self.beta = nn.Parameter(torch.zeros(self.L, self.K, self.G))
        self.log_psi = nn.Parameter(torch.zeros(1, self.G))

    def forward(self, s, omega):
        h = self.basis_gen.compute_h(s)
        spatial_part = torch.einsum("nl,lkg->nkg", h, self.beta)
        eta = self.theta + spatial_part
        log_mu = torch.sum(omega.unsqueeze(-1) * eta, dim=1)
        return torch.exp(log_mu)

    def get_psi(self):
        return torch.exp(self.log_psi)


def nb_loss_mean(model, data_batch):
    mu = model(data_batch["s"], data_batch["omega"])
    psi = model.get_psi()
    denom = psi + mu + 1e-8
    probs = torch.clamp(psi / denom, min=1e-6, max=1.0 - 1e-6)
    dist = torch.distributions.NegativeBinomial(total_count=psi, probs=probs)
    return -dist.log_prob(data_batch["y"]).mean()


def soft_threshold_adaptive(w, base_lambda, v_hat, epsilon=1e-8):
    adaptive_threshold = base_lambda / (torch.sqrt(v_hat) + epsilon)
    return torch.sign(w) * torch.maximum(torch.abs(w) - adaptive_threshold, torch.zeros_like(w))


def clone_state(state):
    return {k: v.detach().clone() for k, v in state.items()}


def local_update(state, data, k, g, basis_gen, l1_lambda, global_delta,
                 local_steps, lr, drift_scale):
    model = CTSVModel(k, g, basis_gen)
    model.load_state_dict(state)
    params = list(model.parameters())
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) + 1e-8 for p in params]
    current_lr, beta1, beta2, eps = lr, 0.9, 0.99, 1e-8
    last_loss = None
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
                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
                m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                v_hat = v[i] / (1 - beta2 ** (epoch + 1))
                step = m_hat / (torch.sqrt(v_hat) + eps)
                drift = drift_scale * global_delta[i] if global_delta is not None else 0.0
                p_temp = p.data - current_lr * (step + drift)
                if p.shape == model.beta.shape:
                    p.data = soft_threshold_adaptive(p_temp, current_lr * l1_lambda, v_hat, eps)
                else:
                    p.data = p_temp
        current_lr *= 0.98
    new_state = clone_state(model.state_dict())
    delta = {key: new_state[key] - state[key] for key in state}
    return new_state, delta, last_loss


def log1p_gene_moments(X):
    """Return per-gene sum and sum of squares for log1p counts."""
    if sparse.issparse(X):
        X_log = X.copy().astype(np.float64)
        X_log.data = np.log1p(X_log.data)
        sums = np.asarray(X_log.sum(axis=0)).ravel()
        sums_sq = np.asarray(X_log.multiply(X_log).sum(axis=0)).ravel()
    else:
        X_log = np.log1p(np.asarray(X, dtype=np.float64))
        sums = X_log.sum(axis=0)
        sums_sq = np.square(X_log).sum(axis=0)
    return sums, sums_sq


def select_shared_high_variance_genes(adatas, gene_limit):
    """Select a shared gene panel by aggregate log1p variance across sections."""
    base_names = list(adatas[0].var_names)
    for adata in adatas[1:]:
        if list(adata.var_names) != base_names:
            raise ValueError("Visium sections have non-identical gene order; explicit gene alignment required.")

    n_genes = adatas[0].shape[1]
    gene_limit_actual = min(gene_limit, n_genes)
    total_n = 0
    total_sum = np.zeros(n_genes, dtype=np.float64)
    total_sumsq = np.zeros(n_genes, dtype=np.float64)
    for adata in adatas:
        sums, sums_sq = log1p_gene_moments(adata.X)
        total_sum += sums
        total_sumsq += sums_sq
        total_n += adata.n_obs

    means = total_sum / total_n
    variances = np.maximum(total_sumsq / total_n - means**2, 0.0)
    selected_idx = np.argsort(variances)[::-1][:gene_limit_actual]
    selected_names = [base_names[i] for i in selected_idx]
    selected_variances = variances[selected_idx]
    return selected_idx, selected_names, selected_variances


def dense_selected_matrix(X, selected_idx):
    """Extract selected genes as a dense float32 matrix."""
    X_sel = X[:, selected_idx]
    if sparse.issparse(X_sel):
        X_sel = X_sel.toarray()
    return np.asarray(X_sel, dtype=np.float32)


def load_section_clients(gene_limit, device):
    """Load 3 mouse brain sections as clients with shared high-variance genes."""
    adatas = []
    for sec_name in SECTIONS:
        print(f"  Loading {sec_name}...")
        adata = sq.datasets.visium(sec_name)
        adata.var_names_make_unique()
        adatas.append(adata)

    selected_idx, gene_names, gene_variances = select_shared_high_variance_genes(adatas, gene_limit)
    print(f"  Selected {len(gene_names)} shared high-variance genes.")
    print("  Top selected genes: " + ", ".join(gene_names[:8]))

    clients = []
    for sec_name, adata in zip(SECTIONS, adatas):
        s_coords = torch.tensor(adata.obsm["spatial"].astype(np.float32), device=device)
        omega_dict = compute_celltype_proportions(adata, CELLTYPE_MARKERS)
        K = len(CELLTYPE_MARKERS)
        omega = np.stack(list(omega_dict.values()), axis=1).astype(np.float32)
        omega = omega / (omega.sum(axis=1, keepdims=True) + 1e-8)
        X_sel = dense_selected_matrix(adata.X, selected_idx)
        clients.append({
            "name": sec_name.split("_")[-1],
            "y": torch.tensor(X_sel, device=device),
            "s": s_coords,
            "omega": torch.tensor(omega.astype(np.float32), device=device),
            "n_spots": adata.n_obs,
        })
    return clients, K, gene_names, selected_idx, gene_variances


def run_federated(clients, k, g, args):
    full_s = torch.cat([c["s"].cpu() for c in clients], dim=0)
    basis_gen = BasisGenerator(full_s)

    model_init = CTSVModel(k, g, basis_gen)
    init_state = clone_state(model_init.state_dict())
    local_means = [torch.log(torch.mean(c["y"].float(), dim=0) + 1.0) for c in clients]
    avg_log_mean = torch.stack(local_means).mean(dim=0)
    init_state["theta"] = avg_log_mean.unsqueeze(0).unsqueeze(0).repeat(1, k, 1).detach().clone()
    global_state = clone_state(init_state)
    param_names = list(model_init.state_dict().keys())
    global_delta = [torch.zeros_like(v) for v in model_init.parameters()]

    history = []
    for round_idx in range(args.rounds):
        losses, deltas = [], []
        for client in clients:
            _, delta, loss = local_update(
                global_state, client, k, g, basis_gen,
                args.l1_lambda, global_delta, args.local_steps, args.lr, args.drift_scale,
            )
            losses.append(loss)
            deltas.append(delta)
        agg_delta = {}
        for key in global_state:
            agg_delta[key] = torch.stack([d[key] for d in deltas]).mean(dim=0)
            global_state[key] = global_state[key] + agg_delta[key]
        global_delta = [agg_delta[name] for name in param_names]
        history.append({
            "round": round_idx,
            "loss": float(np.mean(losses)),
            "update_norm": float(math.sqrt(sum(
                torch.sum(v ** 2).item() for v in agg_delta.values()
            ))),
        })
        if round_idx % max(1, args.print_every) == 0:
            print(f"    round {round_idx:4d}  loss={np.mean(losses):.4f}")
    return global_state, history


def extract_top_genes(global_state, clients, k, gene_names, gene_indices=None):
    """Extract top CTSV genes per cell type from learned beta."""
    beta = global_state["beta"].detach().cpu().numpy()
    theta = global_state["theta"].detach().cpu().numpy()
    results = []
    L, K, G = beta.shape
    for kk in range(K):
        for gg in range(G):
            z1 = beta[0, kk, gg]
            z2 = beta[1, kk, gg]
            abs_z_max = max(abs(z1), abs(z2))
            results.append({
                "Cell_Type": kk,
                "Cell_Type_Name": list(CELLTYPE_MARKERS.keys())[kk],
                "Gene": gene_names[gg] if gg < len(gene_names) else f"Gene_{gg}",
                "Gene_ID": gg,
                "Original_Gene_ID": int(gene_indices[gg]) if gene_indices is not None and gg < len(gene_indices) else gg,
                "Beta_h1": z1, "Beta_h2": z2,
                "Abs_Beta_Max": abs_z_max,
                "Theta": theta[0, kk, gg],
            })
    df = pd.DataFrame(results)
    df = df.sort_values("Abs_Beta_Max", ascending=False)
    return df


def draw_summary_figure(history, ranking, out_dir):
    """Create a simple summary figure."""
    w, h = 1200, 700
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        title_font = label_font = small_font = ImageFont.load_default()

    draw.text((40, 30), "Mouse Brain Visium — Exploratory AP-FedAdam CTSV", fill=(20, 20, 20), font=title_font)
    draw.text((40, 68), f"3 sections (Ant/Post/Ant2), {len(CELLTYPE_MARKERS)} cell types, "
              f"{len(history)} rounds, shared high-variance panel", fill=(80, 80, 80), font=label_font)

    left, top, pw, ph = 100, 120, 450, 350
    losses = [h["loss"] for h in history]
    ymin, ymax = 0.0, max(losses) * 1.1
    for tick in np.linspace(0, ymax, 5):
        y = top + ph - tick / ymax * ph
        draw.line([(left, y), (left + pw, y)], fill=(225, 225, 225, 255))
        draw.text((left - 60, y - 8), f"{tick:.1f}", fill=(40, 40, 40), font=small_font)
    max_round = len(history) - 1
    points = [(left + r / max(1, max_round) * pw,
               top + ph - l / (ymax + 1e-8) * ph) for r, l in enumerate(losses)]
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180, 255), width=3)
    draw.rectangle([left, top, left + pw, top + ph], outline=(20, 20, 20), width=2)
    draw.text((left + pw // 2 - 30, top + ph + 15), "Round", fill=(20, 20, 20), font=label_font)
    draw.text((15, top + ph // 2 - 20), "Loss", fill=(20, 20, 20), font=label_font)

    tx, ty = 610, 140
    top20 = ranking.head(20)
    draw.text((tx, ty - 40), "Top coefficient-ranked pairs (by |beta| max)", fill=(20, 20, 20), font=label_font)
    headers = ["Gene", "Cell Type", "|beta| max", "Beta h1", "Beta h2"]
    col_widths = [95, 140, 85, 80, 80]
    for j, hdr in enumerate(headers):
        x = tx + sum(col_widths[:j])
        draw.text((x, ty), hdr, fill=(40, 40, 40), font=small_font)
    for i, (_, row) in enumerate(top20.iterrows()):
        yy = ty + 22 + i * 20
        vals = [
            str(row["Gene"])[:12],
            str(row["Cell_Type_Name"])[:18],
            f"{row['Abs_Beta_Max']:.4f}",
            f"{row['Beta_h1']:.4f}",
            f"{row['Beta_h2']:.4f}",
        ]
        for j, val in enumerate(vals):
            x = tx + sum(col_widths[:j])
            draw.text((x, yy), val, fill=(20, 20, 20), font=small_font)

    out_path = out_dir / "fig5_mouse_brain_ctsv_summary.png"
    img.save(out_path)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="AP-FedAdam on mouse brain Visium")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-steps", type=int, default=5)
    p.add_argument("--gene-limit", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l1-lambda", type=float, default=5e-6)
    p.add_argument("--drift-scale", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--print-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Mouse Brain AP-FedAdam: {args.rounds} rounds, {args.gene_limit} genes, "
          f"{args.local_steps} local steps, lambda={args.l1_lambda:g}")
    print("Loading sections...")
    clients, K, gene_names, selected_idx, gene_variances = load_section_clients(args.gene_limit, args.device)
    G = len(gene_names)
    print(f"  {len(clients)} clients, K={K} cell types, G={G} genes")
    for c in clients:
        print(f"    {c['name']}: {c['n_spots']} spots")

    selected_df = pd.DataFrame({
        "Rank": np.arange(1, G + 1),
        "Gene": gene_names,
        "Original_Gene_ID": selected_idx.astype(int),
        "Aggregate_Log1p_Variance": gene_variances,
    })
    selected_df.to_csv(OUT_DIR / "mouse_brain_selected_genes.csv", index=False)

    print("Running federated AP-FedAdam...")
    t0 = time.time()
    global_state, history = run_federated(clients, K, G, args)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s, final loss={history[-1]['loss']:.4f}")

    ranking = extract_top_genes(global_state, clients, K, gene_names, selected_idx)
    ranking.to_csv(OUT_DIR / "mouse_brain_ctsv_ranking.csv", index=False)

    print("\n=== Top 10 CTSV genes ===")
    top = ranking.head(10)
    for _, row in top.iterrows():
        print(f"  {row['Gene']:15s} {row['Cell_Type_Name']:18s} "
              f"|beta|={row['Abs_Beta_Max']:.4f}  beta=({row['Beta_h1']:.4f}, {row['Beta_h2']:.4f})")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_DIR / "mouse_brain_ctsv_history.csv", index=False)
    draw_summary_figure(history, ranking, OUT_DIR)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  mouse_brain_selected_genes.csv")
    print(f"  mouse_brain_ctsv_ranking.csv")
    print(f"  mouse_brain_ctsv_history.csv")
    print(f"  fig5_mouse_brain_ctsv_summary.png")


if __name__ == "__main__":
    main()
