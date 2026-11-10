#!/usr/bin/env python3
"""Run the three-condition synthetic CTSV benchmark and make Figures 1--4.

The benchmark has three goals:
1. recover sparse spatial coefficients when all clients have complete inputs;
2. retain recovery when one client lacks two cell-type proportion columns;
3. quantify sensitivity when one client uses a rotated coordinate frame.

Only AP-FedAdam and FedAvg are compared.  AP-FedAdam uses local Adam,
adaptive proximal thresholding for the spatial block, and the previous global
update as a bounded drift direction.  FedAvg uses the same local-step budget
with SGD and weighted server averaging.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "overleaf" / "figs" / "simulation_robustness"
SOURCE_DIR = FIG_DIR / "source_data"

PALETTE = {
    "ap": "#0F4D92",
    "fedavg": "#B64342",
    "neutral": "#727272",
    "light": "#D8D8D8",
    "missing": "#B7B7B7",
    "available": "#4D8AC6",
    "target": "#E28E2C",
}

METHODS = {
    "AP-FedAdam": {"adaptive": True, "prox": True, "drift": True, "color": PALETTE["ap"]},
    "FedAvg": {"adaptive": False, "prox": False, "drift": False, "color": PALETTE["fedavg"]},
}


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    missing_types: tuple[int, ...] = ()
    rotation_deg: float = 0.0


CONDITIONS = (
    Condition("complete", "Complete"),
    Condition("missing", "Client 2 missing CT2 + CT4", missing_types=(1, 3)),
    Condition("rotation30", "Client 2 rotated 30 degrees", rotation_deg=30.0),
    Condition("rotation60", "Client 2 rotated 60 degrees", rotation_deg=60.0),
    Condition("rotation90", "Client 2 rotated 90 degrees", rotation_deg=90.0),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def compute_basis(coords: torch.Tensor) -> torch.Tensor:
    """Two fixed, globally scaled linear bases in the canonical coordinate system."""
    return math.sqrt(12.0) * (coords - 0.5)


class CTSVModel(nn.Module):
    def __init__(self, n_cell_types: int, n_genes: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1, n_cell_types, n_genes))
        self.beta = nn.Parameter(torch.zeros(2, n_cell_types, n_genes))
        self.log_psi = nn.Parameter(torch.full((1, n_genes), math.log(10.0)))

    def forward(self, coords: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        basis = compute_basis(coords)
        spatial = torch.einsum("nl,lkg->nkg", basis, self.beta)
        log_mu = torch.sum(omega.unsqueeze(-1) * (self.theta + spatial), dim=1)
        return torch.exp(torch.clamp(log_mu, min=-7.0, max=6.0))

    def dispersion(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_psi), min=0.2, max=100.0)


def nb_negative_log_likelihood(mu: torch.Tensor, psi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean negative NB log likelihood with E[Y]=mu and dispersion psi."""
    denom = psi + mu + 1e-8
    log_prob = (
        torch.lgamma(y + psi)
        - torch.lgamma(psi)
        - torch.lgamma(y + 1.0)
        + psi * (torch.log(psi + 1e-8) - torch.log(denom))
        + y * (torch.log(mu + 1e-8) - torch.log(denom))
    )
    return -log_prob.mean()


def make_sparse_coefficients(n_genes: int, n_cell_types: int, generator: torch.Generator) -> torch.Tensor:
    beta = torch.zeros(2, n_cell_types, n_genes)
    active_per_type = max(12, n_genes // 12)
    for cell_type in range(n_cell_types):
        candidates = torch.randperm(n_genes, generator=generator)[:active_per_type]
        magnitude = 0.70 + 0.45 * torch.rand(2, active_per_type, generator=generator)
        sign = torch.where(
            torch.rand(2, active_per_type, generator=generator) > 0.5,
            torch.ones(2, active_per_type),
            -torch.ones(2, active_per_type),
        )
        beta[:, cell_type, candidates] = magnitude * sign
    return beta


def generate_synthetic_clients(
    seed: int,
    n_clients: int,
    spots_per_client: int,
    n_genes: int,
    n_cell_types: int,
    device: str,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Generate an aligned, correctly specified NB CTSV data set."""
    set_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    theta = 0.75 + 0.15 * torch.randn(1, n_cell_types, n_genes, generator=generator)
    true_beta = make_sparse_coefficients(n_genes, n_cell_types, generator)
    concentration = torch.full((n_cell_types,), 0.65)
    psi = torch.full((1, n_genes), 12.0)
    clients: list[dict[str, Any]] = []

    for client_id in range(n_clients):
        coords = torch.rand(spots_per_client, 2, generator=generator)
        omega = torch.distributions.Dirichlet(concentration).sample((spots_per_client,))
        basis = compute_basis(coords)
        spatial = torch.einsum("nl,lkg->nkg", basis, true_beta)
        log_mu = torch.sum(omega.unsqueeze(-1) * (theta + spatial), dim=1)
        mu = torch.exp(torch.clamp(log_mu, min=-4.0, max=4.0))
        probs = torch.clamp(mu / (mu + psi), min=1e-6, max=1.0 - 1e-6)
        y = torch.distributions.NegativeBinomial(total_count=psi, probs=probs).sample()
        clients.append(
            {
                "id": client_id,
                "y": y.to(device=device, dtype=torch.float32),
                "coords": coords.to(device=device, dtype=torch.float32),
                "omega": omega.to(device=device, dtype=torch.float32),
                "n_spots": spots_per_client,
            }
        )
    return clients, true_beta.to(device=device, dtype=torch.float32)


def rotate_coordinates(coords: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if angle_deg == 0.0:
        return coords.detach().clone()
    angle = math.radians(angle_deg)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=coords.dtype,
        device=coords.device,
    )
    center = torch.tensor([0.5, 0.5], dtype=coords.dtype, device=coords.device)
    return (coords - center) @ rotation.T + center


def apply_condition(
    base_clients: list[dict[str, Any]],
    condition: Condition,
    n_cell_types: int,
    target_client: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Mask cell types or rotate fitted coordinates while preserving raw counts."""
    active = torch.ones(len(base_clients), n_cell_types, device=base_clients[0]["y"].device)
    clients: list[dict[str, Any]] = []
    for index, client in enumerate(base_clients):
        coords = client["coords"].detach().clone()
        omega = client["omega"].detach().clone()
        if index == target_client and condition.rotation_deg:
            coords = rotate_coordinates(coords, condition.rotation_deg)
        if index == target_client and condition.missing_types:
            omega[:, list(condition.missing_types)] = 0.0
            omega = omega / omega.sum(dim=1, keepdim=True).clamp_min(1e-8)
            active[index, list(condition.missing_types)] = 0.0
        clients.append(
            {
                "id": client["id"],
                "y": client["y"].detach().clone(),
                "coords": coords,
                "omega": omega,
                "n_spots": client["n_spots"],
            }
        )
    return clients, active


def initialize_state(clients: list[dict[str, Any]], n_cell_types: int, n_genes: int, device: str) -> dict[str, torch.Tensor]:
    model = CTSVModel(n_cell_types, n_genes).to(device)
    state = clone_state(model.state_dict())
    all_y = torch.cat([client["y"] for client in clients], dim=0)
    mean_log = torch.log(all_y.mean(dim=0) + 0.2)
    state["theta"] = mean_log.view(1, 1, -1).repeat(1, n_cell_types, 1)
    return state


def local_update(
    state: dict[str, torch.Tensor],
    client: dict[str, Any],
    n_cell_types: int,
    n_genes: int,
    config: dict[str, Any],
    local_steps: int,
    lr: float,
    l1_lambda: float,
    drift: dict[str, torch.Tensor] | None,
    drift_scale: float,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    model = CTSVModel(n_cell_types, n_genes).to(device)
    model.load_state_dict(state)
    params = list(model.named_parameters())
    first_moment = {name: torch.zeros_like(param) for name, param in params}
    second_moment = {name: torch.zeros_like(param) for name, param in params}
    beta1, beta2, eps = 0.9, 0.99, 1e-8
    last_loss = float("nan")

    for step_index in range(local_steps):
        model.zero_grad(set_to_none=True)
        loss = nb_negative_log_likelihood(model(client["coords"], client["omega"]), model.dispersion(), client["y"])
        loss.backward()
        last_loss = float(loss.detach().cpu())
        step_lr = lr * (0.98**step_index)
        with torch.no_grad():
            for name, parameter in params:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if config["adaptive"]:
                    first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
                    second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * gradient.square()
                    m_hat = first_moment[name] / (1.0 - beta1 ** (step_index + 1))
                    v_hat = second_moment[name] / (1.0 - beta2 ** (step_index + 1))
                    direction = m_hat / (v_hat.sqrt() + eps)
                else:
                    v_hat = torch.ones_like(parameter)
                    direction = gradient
                if config["drift"] and drift is not None:
                    direction = direction + drift_scale * drift[name]
                proposal = parameter - step_lr * direction
                if config["prox"] and name == "beta":
                    threshold = step_lr * l1_lambda / (v_hat.sqrt() + eps)
                    parameter.copy_(torch.sign(proposal) * torch.clamp(proposal.abs() - threshold, min=0.0))
                else:
                    parameter.copy_(proposal)
                parameter.nan_to_num_(nan=0.0, posinf=25.0, neginf=-25.0)

    updated_state = clone_state(model.state_dict())
    delta = {name: updated_state[name] - state[name] for name in state}
    return updated_state, delta, last_loss


def aggregate_deltas(
    deltas: list[dict[str, torch.Tensor]],
    clients: list[dict[str, Any]],
    active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    total_spots = float(sum(client["n_spots"] for client in clients))
    weights = torch.tensor([client["n_spots"] / total_spots for client in clients], device=active.device)
    aggregate: dict[str, torch.Tensor] = {}
    for name in deltas[0]:
        stacked = torch.stack([delta[name] for delta in deltas], dim=0)
        if name == "theta":
            block_weight = weights[:, None, None, None] * active[:, None, :, None]
            numerator = (stacked * block_weight).sum(dim=0)
            denominator = block_weight.sum(dim=0)
            aggregate[name] = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-10), torch.zeros_like(numerator))
        elif name == "beta":
            block_weight = weights[:, None, None, None] * active[:, None, :, None]
            numerator = (stacked * block_weight).sum(dim=0)
            denominator = block_weight.sum(dim=0)
            aggregate[name] = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-10), torch.zeros_like(numerator))
        else:
            aggregate[name] = (stacked * weights.view(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)
    return aggregate


def update_norm(delta: dict[str, torch.Tensor]) -> float:
    return float(torch.sqrt(sum(torch.sum(value.square()) for value in delta.values())).detach().cpu())


def coefficient_metrics(estimate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    predicted = estimate.detach().cpu().numpy().ravel()
    target = truth.detach().cpu().numpy().ravel()
    difference = predicted - target
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "pearson": float(np.corrcoef(predicted, target)[0, 1]),
        "support_recall": float(
            np.mean((np.abs(predicted[np.abs(target) > 1e-8]) > 0.10))
        ),
    }


def run_federated(
    clients: list[dict[str, Any]],
    active: torch.Tensor,
    truth: torch.Tensor,
    method: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, torch.Tensor]]:
    config = METHODS[method]
    state = initialize_state(clients, args.cell_types, args.genes, args.device)
    drift = {name: torch.zeros_like(value) for name, value in state.items()}
    learning_rate = args.ap_lr if method == "AP-FedAdam" else args.fedavg_lr
    history: list[dict[str, float]] = []
    for round_index in range(args.rounds):
        deltas = []
        losses = []
        for client in clients:
            _, delta, loss = local_update(
                state,
                client,
                args.cell_types,
                args.genes,
                config,
                args.local_steps,
                learning_rate,
                args.l1_lambda,
                drift,
                args.drift_scale,
                args.device,
            )
            deltas.append(delta)
            losses.append(loss)
        aggregate = aggregate_deltas(deltas, clients, active)
        state = {name: state[name] + aggregate[name] for name in state}
        drift = clone_state(aggregate)
        history.append(
            {
                "round": round_index + 1,
                "loss": float(np.mean(losses)),
                "update_norm": update_norm(aggregate),
            }
        )
    metrics = coefficient_metrics(state["beta"], truth)
    summary = {
        "method": method,
        "final_loss": history[-1]["loss"],
        "final_update_norm": history[-1]["update_norm"],
        **metrics,
    }
    return pd.DataFrame(history), summary, state


def choose_candidate_genes(truth: torch.Tensor, n_candidates: int = 5) -> list[int]:
    values = truth.detach().cpu().numpy()
    candidates: list[int] = []
    for cell_type in range(values.shape[1]):
        score = np.linalg.norm(values[:, cell_type, :], axis=0)
        candidates.append(int(np.argmax(score)))
    if len(set(candidates)) < n_candidates:
        global_score = np.linalg.norm(values, axis=(0, 1))
        for gene in np.argsort(global_score)[::-1]:
            if int(gene) not in candidates:
                candidates.append(int(gene))
            if len(candidates) == n_candidates:
                break
    return candidates[:n_candidates]


def coefficient_rows(
    condition: str,
    method: str,
    repeat: int,
    truth: torch.Tensor,
    state: dict[str, torch.Tensor],
    candidates: list[int],
) -> list[dict[str, Any]]:
    estimate = state["beta"].detach().cpu().numpy()
    target = truth.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for gene in candidates:
        for basis in range(2):
            for cell_type in range(target.shape[1]):
                rows.append(
                    {
                        "condition": condition,
                        "method": method,
                        "repeat": repeat,
                        "gene": gene,
                        "basis": basis + 1,
                        "cell_type": cell_type + 1,
                        "true_beta": float(target[basis, cell_type, gene]),
                        "learned_beta": float(estimate[basis, cell_type, gene]),
                    }
                )
    return rows


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def coefficient_matrix(values: np.ndarray, gene: int) -> np.ndarray:
    return values[:, :, gene]


def draw_coefficient_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vlim: float,
    show_y: bool,
    show_x: bool,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title, pad=3, fontsize=6.4, fontweight="bold")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"CT{k + 1}" for k in range(matrix.shape[1])] if show_x else [])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["h1", "h2"] if show_y else [])
    ax.tick_params(length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    return image


def draw_training_curve(ax: plt.Axes, history: pd.DataFrame, column: str, title: str, log_scale: bool = False) -> None:
    ax.plot(history["round"], history[column], color=PALETTE["ap"], linewidth=1.3)
    ax.scatter(history["round"].iloc[-1], history[column].iloc[-1], color=PALETTE["ap"], s=9, zorder=3)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("NB loss" if column == "loss" else "Update norm")
    ax.set_xlim(1, int(history["round"].max()))


def export_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 450},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_complete_figure(artifact: dict[str, Any], candidates: list[int]) -> None:
    truth = artifact["truth"].detach().cpu().numpy()
    learned = artifact["state"]["beta"].detach().cpu().numpy()
    vlim = max(np.abs(truth[:, :, candidates]).max(), np.abs(learned[:, :, candidates]).max()) * 1.05
    fig = plt.figure(figsize=(10.8, 3.8))
    grid = fig.add_gridspec(2, 6, width_ratios=[1.30, 1, 1, 1, 1, 1], hspace=0.38, wspace=0.45)
    ax_loss = fig.add_subplot(grid[0, 0])
    draw_training_curve(ax_loss, artifact["history"], "loss", "Training loss")
    panel_label(ax_loss, "a")
    ax_norm = fig.add_subplot(grid[1, 0])
    draw_training_curve(ax_norm, artifact["history"], "update_norm", "Update norm", log_scale=True)

    image = None
    for col, gene in enumerate(candidates, start=1):
        top = fig.add_subplot(grid[0, col])
        image = draw_coefficient_heatmap(top, coefficient_matrix(truth, gene), f"G{gene}: true", vlim, col == 1, False)
        bottom = fig.add_subplot(grid[1, col])
        draw_coefficient_heatmap(bottom, coefficient_matrix(learned, gene), f"G{gene}: learned", vlim, col == 1, True)
    colorbar = fig.colorbar(image, ax=fig.axes[2:], fraction=0.016, pad=0.012)
    colorbar.set_label("Spatial coefficient", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    fig.suptitle("Complete-client simulation: AP-FedAdam recovers sparse spatial coefficients", x=0.50, y=1.02, fontsize=8.5, fontweight="bold")
    export_figure(fig, "fig1_complete_recovery")


def draw_missing_figure(artifact: dict[str, Any], candidates: list[int], n_clients: int, n_cell_types: int) -> None:
    truth = artifact["truth"].detach().cpu().numpy()
    learned = artifact["state"]["beta"].detach().cpu().numpy()
    vlim = max(np.abs(truth[:, :, candidates]).max(), np.abs(learned[:, :, candidates]).max()) * 1.05
    fig = plt.figure(figsize=(10.8, 3.8))
    grid = fig.add_gridspec(2, 6, width_ratios=[1.30, 1, 1, 1, 1, 1], hspace=0.38, wspace=0.45)
    availability = np.ones((n_clients, n_cell_types))
    availability[1, [1, 3]] = 0.0
    ax_availability = fig.add_subplot(grid[0, 0])
    panel_label(ax_availability, "a")
    ax_availability.imshow(availability, cmap=mpl.colors.ListedColormap([PALETTE["missing"], PALETTE["available"]]), vmin=0, vmax=1, aspect="auto")
    ax_availability.set_xticks(range(n_cell_types))
    ax_availability.set_xticklabels([f"CT{k + 1}" for k in range(n_cell_types)])
    ax_availability.set_yticks(range(n_clients))
    ax_availability.set_yticklabels([f"C{k + 1}" for k in range(n_clients)])
    ax_availability.set_title("Observed cell-type panel", loc="left", pad=3, fontweight="bold")
    ax_availability.tick_params(length=0, pad=1)
    ax_availability.text(1.0, 1.0, "missing", ha="center", va="center", color="white", fontsize=5.5, fontweight="bold")
    ax_availability.text(3.0, 1.0, "missing", ha="center", va="center", color="white", fontsize=5.5, fontweight="bold")

    ax_loss = fig.add_subplot(grid[1, 0])
    draw_training_curve(ax_loss, artifact["history"], "loss", "Training loss")
    image = None
    for col, gene in enumerate(candidates, start=1):
        top = fig.add_subplot(grid[0, col])
        image = draw_coefficient_heatmap(top, coefficient_matrix(truth, gene), f"G{gene}: true", vlim, col == 1, False)
        bottom = fig.add_subplot(grid[1, col])
        draw_coefficient_heatmap(bottom, coefficient_matrix(learned, gene), f"G{gene}: learned", vlim, col == 1, True)
    colorbar = fig.colorbar(image, ax=fig.axes[2:], fraction=0.016, pad=0.012)
    colorbar.set_label("Spatial coefficient", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    fig.suptitle("Incomplete-client simulation: block-wise federation retains coefficient recovery", x=0.53, y=1.02, fontsize=8.5, fontweight="bold")
    export_figure(fig, "fig2_missing_celltypes")


def plot_coordinate_panel(ax: plt.Axes, base: torch.Tensor, rotated: torch.Tensor, title: str, color: str) -> None:
    ax.scatter(base[:, 0].cpu(), base[:, 1].cpu(), s=4, color="#D4D4D4", linewidths=0, alpha=0.65)
    ax.scatter(rotated[:, 0].cpu(), rotated[:, 1].cpu(), s=4, color=color, linewidths=0, alpha=0.70)
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, loc="left", pad=3, fontweight="bold")


def draw_rotation_figure(summary: pd.DataFrame, artifact: dict[str, Any], candidates: list[int]) -> None:
    representative = artifact["base_clients"][1]["coords"]
    rotated = rotate_coordinates(representative, 90.0)
    truth = artifact["truth"].detach().cpu().numpy()
    learned = artifact["state"]["beta"].detach().cpu().numpy()
    vlim = max(np.abs(truth[:, :, candidates]).max(), np.abs(learned[:, :, candidates]).max()) * 1.05
    fig = plt.figure(figsize=(10.8, 5.4))
    grid = fig.add_gridspec(2, 8, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.75)

    ax_original = fig.add_subplot(grid[0, 0:2])
    panel_label(ax_original, "a")
    plot_coordinate_panel(ax_original, representative, representative, "Client 2: aligned", PALETTE["target"])
    ax_rotated = fig.add_subplot(grid[0, 2:4])
    panel_label(ax_rotated, "b")
    plot_coordinate_panel(ax_rotated, representative, rotated, "Client 2: 90 degree rotation", PALETTE["target"])

    ax_curve = fig.add_subplot(grid[0, 5:8])
    panel_label(ax_curve, "c", x=-0.20)
    curve = summary[(summary["method"] == "AP-FedAdam") & (summary["condition"].isin(["complete", "rotation30", "rotation60", "rotation90"]))].copy()
    curve["angle"] = curve["rotation_deg"].fillna(0.0)
    grouped = curve.groupby("angle", as_index=False).agg(mean=("mae", "mean"), sd=("mae", "std"))
    grouped["sd"] = grouped["sd"].fillna(0.0)
    ax_curve.errorbar(grouped["angle"], grouped["mean"], yerr=grouped["sd"], color=PALETTE["ap"], marker="o", linewidth=1.4, markersize=4, capsize=2)
    ax_curve.set_xticks([0, 30, 60, 90])
    ax_curve.set_xlabel("Rotation angle of Client 2 (degrees)")
    ax_curve.set_ylabel("Coefficient MAE")
    ax_curve.set_title("Coordinate-frame dependence", loc="left", pad=3, fontweight="bold")
    ax_curve.text(0.02, 0.96, f"mean +/- s.d.; n={curve['repeat'].nunique()} simulations", transform=ax_curve.transAxes, va="top", fontsize=5.5, color=PALETTE["neutral"])

    bottom_grid = grid[1, :].subgridspec(1, len(candidates), wspace=0.55)
    image = None
    for index, gene in enumerate(candidates):
        axis = fig.add_subplot(bottom_grid[0, index])
        image = draw_coefficient_heatmap(axis, coefficient_matrix(learned, gene) - coefficient_matrix(truth, gene), f"G{gene}: learned - true", vlim, index == 0, True)
    colorbar = fig.colorbar(image, ax=fig.axes[3:], fraction=0.016, pad=0.015)
    colorbar.set_label("Coefficient error", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    fig.suptitle("Coordinate mismatch degrades recovery when one federated client is not spatially registered", x=0.51, y=1.01, fontsize=8.5, fontweight="bold")
    export_figure(fig, "fig3_coordinate_rotation")


def draw_fedavg_comparison(summary: pd.DataFrame, histories: pd.DataFrame) -> None:
    figure_conditions = ["complete", "missing", "rotation90"]
    labels = ["Complete", "Missing CT", "Rotation 90"]
    fig = plt.figure(figsize=(7.25, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.58, wspace=0.45)
    ax_bar = fig.add_subplot(grid[0, :])
    panel_label(ax_bar, "a", x=-0.07)
    subset = summary[summary["condition"].isin(figure_conditions)].copy()
    aggregate = subset.groupby(["condition", "method"], as_index=False).agg(mean=("mae", "mean"), sd=("mae", "std"))
    aggregate["sd"] = aggregate["sd"].fillna(0.0)
    x = np.arange(len(figure_conditions))
    width = 0.34
    for method_index, method in enumerate(["AP-FedAdam", "FedAvg"]):
        values = aggregate[aggregate["method"] == method].set_index("condition").loc[figure_conditions]
        offset = (method_index - 0.5) * width
        bars = ax_bar.bar(x + offset, values["mean"], width=width, yerr=values["sd"], capsize=2.5, color=METHODS[method]["color"], edgecolor="black", linewidth=0.45, label=method)
        for bar, value in zip(bars, values["mean"]):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(aggregate["mean"]) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Coefficient MAE")
    ax_bar.set_title("AP-FedAdam versus tuned FedAvg", loc="left", pad=4, fontweight="bold")
    ax_bar.legend(ncol=2, loc="upper left")

    for panel_index, (condition, label) in enumerate(zip(figure_conditions, labels)):
        ax = fig.add_subplot(grid[1, panel_index])
        panel_label(ax, chr(ord("b") + panel_index), x=-0.20)
        view = histories[(histories["condition"] == condition) & (histories["repeat"] == 0)]
        for method in ["AP-FedAdam", "FedAvg"]:
            line = view[view["method"] == method].sort_values("round")
            ax.plot(line["round"], line["loss"], color=METHODS[method]["color"], linewidth=1.25, label=method)
        ax.set_title(label, loc="left", pad=3, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("NB loss" if panel_index == 0 else "")
        ax.set_xlim(1, int(view["round"].max()))
    fig.suptitle("Federated optimizer comparison across complete, incomplete and rotated-coordinate clients", x=0.50, y=1.01, fontsize=8.5, fontweight="bold")
    export_figure(fig, "fig4_apfedadam_vs_fedavg")


def write_qa_notes(args: argparse.Namespace, candidates: list[int]) -> None:
    note = [
        "# Simulation Figures QA Notes",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: AP-FedAdam recovers sparse CTSV coefficients under complete client inputs and retains lower error than tuned FedAvg when a client has incomplete cell-type inputs or a mismatched coordinate frame.",
        "- Figure archetype: quantitative grid with coefficient-recovery heatmaps.",
        "- Backend: Python/matplotlib only.",
        "- Source-data note: Figure 4 bars summarize independent synthetic data sets; curves and coefficient maps use the first deterministic replicate.",
        "- Image-integrity note: all image panels are direct matrix or coordinate-scatter renderings from the saved source CSV files; no local image adjustment was applied.",
        "",
        "## Simulation Setting",
        "",
        f"- Clients: {args.clients}; spots per client: {args.spots_per_client}; genes: {args.genes}; cell types: {args.cell_types}.",
        f"- Communication rounds: {args.rounds}; local steps: {args.local_steps}; independent data simulations: {args.repeats}.",
        f"- Candidate genes shown in coefficient heatmaps: {', '.join(map(str, candidates))}.",
        "- Missing-input condition: Client 2 does not provide CT2 or CT4, and server aggregation excludes that client for the corresponding theta/beta blocks.",
        "- Rotation condition: counts are generated in the canonical frame; Client 2 is fitted after a coordinate rotation, so the experiment measures sensitivity to spatial misregistration.",
        "",
        "## Exports",
        "",
        "- SVG has editable text (svg.fonttype=none); PDF uses TrueType text; TIFF is exported at 600 dpi.",
    ]
    (FIG_DIR / "simulation_figures_qa_notes.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-condition AP-FedAdam synthetic CTSV benchmark.")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--spots-per-client", type=int, default=100)
    parser.add_argument("--genes", type=int, default=160)
    parser.add_argument("--cell-types", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--ap-lr", type=float, default=0.035)
    parser.add_argument("--fedavg-lr", type=float, default=0.035)
    parser.add_argument("--l1-lambda", type=float, default=2e-4)
    parser.add_argument("--drift-scale", type=float, default=0.08)
    parser.add_argument("--target-client", type=int, default=1, help="Zero-based index of the incomplete/rotated client.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_client < 0 or args.target_client >= args.clients:
        raise ValueError("target-client must index an existing client")
    if args.cell_types != 5:
        raise ValueError("The current figure layout expects five cell types.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    history_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_genes: list[int] | None = None

    for repeat in range(args.repeats):
        data_seed = args.seed + repeat
        base_clients, truth = generate_synthetic_clients(
            data_seed,
            args.clients,
            args.spots_per_client,
            args.genes,
            args.cell_types,
            args.device,
        )
        if candidate_genes is None:
            candidate_genes = choose_candidate_genes(truth)
        for condition_index, condition in enumerate(CONDITIONS):
            clients, active = apply_condition(base_clients, condition, args.cell_types, args.target_client)
            for method_index, method in enumerate(METHODS):
                set_seed(data_seed * 101 + condition_index * 11 + method_index)
                history, summary, state = run_federated(clients, active, truth, method, args)
                history["repeat"] = repeat
                history["seed"] = data_seed
                history["condition"] = condition.key
                history["method"] = method
                history_rows.append(history)
                summary_rows.append(
                    {
                        **summary,
                        "repeat": repeat,
                        "seed": data_seed,
                        "condition": condition.key,
                        "condition_label": condition.label,
                        "rotation_deg": condition.rotation_deg,
                        "missing_cell_types": ",".join(str(value + 1) for value in condition.missing_types),
                    }
                )
                if repeat == 0 and method == "AP-FedAdam" and condition.key in {"complete", "missing", "rotation90"}:
                    artifacts[(condition.key, method)] = {
                        "history": history.copy(),
                        "state": state,
                        "truth": truth,
                        "base_clients": base_clients,
                    }
                if repeat == 0 and method == "AP-FedAdam":
                    beta_rows.extend(coefficient_rows(condition.key, method, repeat, truth, state, candidate_genes))
                print(
                    f"repeat={repeat + 1}/{args.repeats} condition={condition.key:10s} method={method:10s} "
                    f"MAE={summary['mae']:.4f} r={summary['pearson']:.3f} loss={summary['final_loss']:.4f}"
                )

    histories = pd.concat(history_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    beta_source = pd.DataFrame(beta_rows)
    histories.to_csv(SOURCE_DIR / "simulation_training_history.csv", index=False)
    summary.to_csv(SOURCE_DIR / "simulation_summary.csv", index=False)
    beta_source.to_csv(SOURCE_DIR / "selected_coefficient_matrices.csv", index=False)
    manifest = {
        "clients": args.clients,
        "spots_per_client": args.spots_per_client,
        "genes": args.genes,
        "cell_types": args.cell_types,
        "rounds": args.rounds,
        "local_steps": args.local_steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "ap_lr": args.ap_lr,
        "fedavg_lr": args.fedavg_lr,
        "l1_lambda": args.l1_lambda,
        "drift_scale": args.drift_scale,
        "candidate_genes": candidate_genes,
        "conditions": [condition.__dict__ for condition in CONDITIONS],
    }
    (SOURCE_DIR / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    draw_complete_figure(artifacts[("complete", "AP-FedAdam")], candidate_genes)
    draw_missing_figure(artifacts[("missing", "AP-FedAdam")], candidate_genes, args.clients, args.cell_types)
    draw_rotation_figure(summary, artifacts[("rotation90", "AP-FedAdam")], candidate_genes)
    draw_fedavg_comparison(summary, histories)
    write_qa_notes(args, candidate_genes)
    print(f"Wrote figures and source data to {FIG_DIR}")


if __name__ == "__main__":
    main()
