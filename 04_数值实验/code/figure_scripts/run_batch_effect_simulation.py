#!/usr/bin/env python3
"""Simulate slice-level batch effects in AP-FedAdam synthetic CTSV data.

The experiment reuses the existing synthetic CTSV benchmark, keeps only a small
set of clients as "slices", and creates a simple batch effect by adding a
constant count offset to selected slices. This is intentionally a stress test:
the model has no explicit client intercept or batch-correction covariate, so a
client-level global shift is treated as federated heterogeneity.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_fig3_ablation import (
    BasisGenerator,
    METHODS,
    METHOD_COLORS,
    load_clients,
    load_true_beta,
    run_method,
    selected_methods,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "overleaf" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_indices(text: str) -> list[int]:
    if not text.strip():
        return []
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def clone_clients_with_shift(clients, shifted_indices: set[int], count_shift: float):
    shifted = []
    for idx, client in enumerate(clients):
        y = client["y"].detach().clone()
        before_mean = float(y.float().mean().item())
        applied_shift = count_shift if idx in shifted_indices else 0.0
        if applied_shift:
            y = y + applied_shift
        shifted.append(
            {
                "id": client["id"],
                "y": y,
                "s": client["s"].detach().clone(),
                "omega": client["omega"].detach().clone(),
                "batch_shift": applied_shift,
                "mean_before": before_mean,
                "mean_after": float(y.float().mean().item()),
            }
        )
    return shifted


def client_mean_rows(scenario: str, clients):
    rows = []
    for idx, client in enumerate(clients):
        rows.append(
            {
                "scenario": scenario,
                "client_index": idx,
                "client_id": client["id"],
                "batch_shift": client["batch_shift"],
                "mean_count_before": client["mean_before"],
                "mean_count_after": client["mean_after"],
            }
        )
    return rows


def tagged(stem: str, tag: str, suffix: str) -> Path:
    clean = tag.replace("/", "_").replace(" ", "_")
    return OUT_DIR / f"{stem}_{clean}{suffix}"


def draw_grouped_mae(summary_df: pd.DataFrame, out_path: Path):
    scenarios = summary_df["scenario"].drop_duplicates().tolist()
    methods = summary_df["method"].drop_duplicates().tolist()
    w, h = 1120, 720
    left, top, plot_w, plot_h = 115, 110, 780, 430
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(31, bold=True)
    label_font = font(22, bold=True)
    small_font = font(18)
    value_font = font(16, bold=True)
    draw.text((38, 35), "Batch-effect simulation: final beta MAE", fill=(20, 20, 20), font=title_font)

    vals = summary_df["final_mae"].to_numpy(dtype=float)
    ymax = float(vals.max()) * 1.25 + 1e-8
    for tick in np.linspace(0, ymax, 5):
        y = top + plot_h - tick / ymax * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225, 255))
        draw.text((left - 78, y - 10), f"{tick:.2f}", fill=(40, 40, 40), font=small_font)
    draw.line([(left, top), (left, top + plot_h)], fill=(20, 20, 20), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(20, 20, 20), width=2)

    group_gap = 70
    group_w = (plot_w - group_gap * (len(scenarios) + 1)) / max(1, len(scenarios))
    bar_gap = 10
    bar_w = (group_w - bar_gap * (len(methods) - 1)) / max(1, len(methods))
    for gi, scenario in enumerate(scenarios):
        gx = left + group_gap + gi * (group_w + group_gap)
        for mi, method in enumerate(methods):
            row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["method"] == method)]
            if row.empty:
                continue
            val = float(row["final_mae"].iloc[0])
            x0 = gx + mi * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = top + plot_h - val / ymax * plot_h
            color = METHOD_COLORS.get(method, (90, 90, 90))
            draw.rectangle([x0, y0, x1, top + plot_h], fill=color + (220,))
            draw.text((x0 + 2, y0 - 23), f"{val:.3f}", fill=(20, 20, 20), font=value_font)
        label = scenario.replace("_", "\n")
        draw.multiline_text((gx + 8, top + plot_h + 18), label, fill=(20, 20, 20), font=small_font, spacing=2)

    lx, ly = left + plot_w + 42, top + 35
    for i, method in enumerate(methods):
        yy = ly + i * 42
        color = METHOD_COLORS.get(method, (90, 90, 90))
        draw.rectangle([lx, yy, lx + 30, yy + 20], fill=color + (220,))
        draw.text((lx + 42, yy - 1), method, fill=(20, 20, 20), font=small_font)
    draw.text((left, top - 34), "MAE", fill=(20, 20, 20), font=label_font)
    img.save(out_path)


def draw_loss_curves(history_df: pd.DataFrame, out_path: Path):
    rows = history_df.copy()
    scenarios = rows["scenario"].drop_duplicates().tolist()
    methods = rows["method"].drop_duplicates().tolist()
    w, h = 1280, 760
    left, top, plot_w, plot_h = 120, 112, 850, 470
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(31, bold=True)
    label_font = font(22, bold=True)
    small_font = font(17)
    draw.text((38, 35), "Batch-effect simulation: training loss", fill=(20, 20, 20), font=title_font)

    vals = rows["loss"].to_numpy(dtype=float)
    ymin = 0.0
    ymax = float(vals[np.isfinite(vals)].max()) * 1.08 + 1e-8
    max_round = int(rows["round"].max())
    for tick in np.linspace(0, max_round, 6):
        x = left + tick / max(1, max_round) * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill=(225, 225, 225, 255))
        draw.text((x - 16, top + plot_h + 13), f"{int(tick)}", fill=(40, 40, 40), font=small_font)
    for tick in np.linspace(ymin, ymax, 5):
        y = top + plot_h - (tick - ymin) / (ymax - ymin + 1e-12) * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225, 255))
        draw.text((left - 84, y - 10), f"{tick:.2f}", fill=(40, 40, 40), font=small_font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(20, 20, 20), width=2)

    dash_patterns = {"no_batch": None, "batch_shift": 16}
    for scenario in scenarios:
        for method in methods:
            sub = rows[(rows["scenario"] == scenario) & (rows["method"] == method)].sort_values("round")
            if sub.empty:
                continue
            points = []
            for _, row in sub.iterrows():
                x = left + int(row["round"]) / max(1, max_round) * plot_w
                y = top + plot_h - (float(row["loss"]) - ymin) / (ymax - ymin + 1e-12) * plot_h
                points.append((x, y))
            color = METHOD_COLORS.get(method, (90, 90, 90)) + (255,)
            dash = dash_patterns.get(scenario)
            if dash is None:
                draw.line(points, fill=color, width=4)
            else:
                for a, b in zip(points[:-1], points[1:]):
                    dist = math.dist(a, b)
                    if dist <= 0:
                        continue
                    steps = max(1, int(dist / dash))
                    for step in range(steps):
                        if step % 2:
                            continue
                        t0 = step / steps
                        t1 = min(1.0, (step + 1) / steps)
                        p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                        p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                        draw.line([p0, p1], fill=color, width=4)

    draw.text((left + plot_w / 2 - 45, top + plot_h + 54), "Round", fill=(20, 20, 20), font=label_font)
    draw.text((left, top - 34), "Loss", fill=(20, 20, 20), font=label_font)

    lx, ly = left + plot_w + 45, top + 20
    draw.text((lx, ly - 34), "Methods", fill=(20, 20, 20), font=label_font)
    for i, method in enumerate(methods):
        yy = ly + i * 38
        color = METHOD_COLORS.get(method, (90, 90, 90)) + (255,)
        draw.line([(lx, yy + 10), (lx + 34, yy + 10)], fill=color, width=5)
        draw.text((lx + 46, yy), method, fill=(20, 20, 20), font=small_font)
    sy = ly + len(methods) * 44 + 20
    draw.text((lx, sy), "Scenario", fill=(20, 20, 20), font=label_font)
    draw.line([(lx, sy + 44), (lx + 46, sy + 44)], fill=(50, 50, 50, 255), width=4)
    draw.text((lx + 58, sy + 33), "no_batch", fill=(20, 20, 20), font=small_font)
    for x in range(lx, lx + 46, 18):
        draw.line([(x, sy + 86), (min(x + 10, lx + 46), sy + 86)], fill=(50, 50, 50, 255), width=4)
    draw.text((lx + 58, sy + 75), "batch_shift", fill=(20, 20, 20), font=small_font)
    img.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple client-level batch-effect simulation.")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--gene-limit", type=int, default=300)
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--shifted-client-indices", default="1,2", help="Zero-based client indices to shift.")
    parser.add_argument("--count-shift", type=float, default=2.0, help="Additive count offset for shifted clients.")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l1-lambda", type=float, default=5e-6)
    parser.add_argument("--drift-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--data-mode", default="cos")
    parser.add_argument("--basis-mode", default="cos2")
    parser.add_argument("--methods", default="AP-FedAdam,No-drift,FedAvg")
    parser.add_argument("--output-tag", default="batch_effect_shift2")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    k = 5
    g = args.gene_limit
    all_clients = load_clients(args.data_mode, g, k, args.device)
    if args.n_clients > len(all_clients):
        raise SystemExit(f"Requested {args.n_clients} clients, but only {len(all_clients)} are available.")
    base_clients = all_clients[: args.n_clients]
    shifted_indices = set(parse_indices(args.shifted_client_indices))
    invalid = [i for i in shifted_indices if i < 0 or i >= len(base_clients)]
    if invalid:
        raise SystemExit(f"Invalid shifted client indices for n_clients={args.n_clients}: {invalid}")

    full_s = torch.cat([client["s"].cpu() for client in base_clients], dim=0)
    basis_gen = BasisGenerator(mode=args.basis_mode, s_coords=full_s)
    true_beta = load_true_beta(args.data_mode, g, k, basis_gen.L, args.device)

    methods = selected_methods(args.methods)
    run_args = SimpleNamespace(**vars(args))

    scenario_defs = [
        ("no_batch", set(), 0.0),
        ("batch_shift", shifted_indices, args.count_shift),
    ]
    all_history = []
    all_summary = []
    all_client_means = []

    print(
        f"Running batch-effect simulation: data={args.data_mode}, basis={args.basis_mode}, "
        f"clients={args.n_clients}, shifted={sorted(shifted_indices)}, count_shift={args.count_shift:g}, "
        f"genes={g}, rounds={args.rounds}, lambda={args.l1_lambda:g}, methods={args.methods}"
    )

    for scenario, scenario_shifted, count_shift in scenario_defs:
        clients = clone_clients_with_shift(base_clients, scenario_shifted, count_shift)
        all_client_means.extend(client_mean_rows(scenario, clients))
        for method_name, config in methods:
            print(f"\nScenario={scenario}, method={method_name}")
            history, summary = run_method(method_name, config, clients, k, g, basis_gen, true_beta, run_args)
            for row in history:
                row["scenario"] = scenario
                row["count_shift"] = count_shift
                row["shifted_client_indices"] = ",".join(map(str, sorted(scenario_shifted)))
            summary["scenario"] = scenario
            summary["count_shift"] = count_shift
            summary["shifted_client_indices"] = ",".join(map(str, sorted(scenario_shifted)))
            summary["n_clients"] = args.n_clients
            all_history.extend(history)
            all_summary.append(summary)
            print(
                f"  final MAE={summary['final_mae']:.4f}, "
                f"Pearson={summary['final_pearson']:.4f}, loss={summary['final_loss']:.4f}"
            )

    history_df = pd.DataFrame(all_history)
    summary_df = pd.DataFrame(all_summary)
    means_df = pd.DataFrame(all_client_means)

    history_path = tagged("batch_effect_history", args.output_tag, ".csv")
    summary_path = tagged("batch_effect_summary", args.output_tag, ".csv")
    means_path = tagged("batch_effect_client_means", args.output_tag, ".csv")
    mae_fig_path = tagged("batch_effect_mae", args.output_tag, ".png")
    loss_fig_path = tagged("batch_effect_loss", args.output_tag, ".png")

    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    means_df.to_csv(means_path, index=False)
    draw_grouped_mae(summary_df, mae_fig_path)
    draw_loss_curves(history_df, loss_fig_path)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {means_path}")
    print(f"Wrote {mae_fig_path}")
    print(f"Wrote {loss_fig_path}")


if __name__ == "__main__":
    main()
