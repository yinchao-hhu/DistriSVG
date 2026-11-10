from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "overleaf" / "figs"
LAMBDA_TAGS = ["1e-6", "5e-6", "1e-5", "5e-5", "1e-4"]


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


def load_summary():
    frames = []
    for tag in LAMBDA_TAGS:
        frames.append(pd.read_csv(FIG_DIR / f"fig3_ablation_summary_stress_lam_{tag}.csv"))
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(FIG_DIR / "fig3_sparsity_stress_summary.csv", index=False)
    wide = summary.pivot(index="l1_lambda", columns="method", values="final_mae").reset_index()
    wide["mae_delta_no_prox_minus_ap"] = wide["No-prox"] - wide["AP-FedAdam"]
    wide.to_csv(FIG_DIR / "fig3_sparsity_stress_mae_wide.csv", index=False)
    return summary


def draw_axes(draw, left, top, plot_w, plot_h, x_logs, x_labels, y_min, y_max, y_formatter):
    small_font = font(18)
    gray = (80, 80, 80)
    for tick in np.linspace(y_min, y_max, 5):
        y = top + plot_h - (tick - y_min) / (y_max - y_min + 1e-12) * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225, 255))
        draw.text((left - 62, y - 10), y_formatter(tick), fill=gray, font=small_font)
    x_span = x_logs.max() - x_logs.min()
    for x_log, label in zip(x_logs, x_labels):
        x = left + (x_log - x_logs.min()) / x_span * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill=(238, 238, 238, 255))
        draw.text((x - 22, top + plot_h + 14), label, fill=gray, font=small_font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(20, 20, 20), width=2)


def draw_line(draw, x_logs, y_vals, left, top, plot_w, plot_h, y_min, y_max, color):
    x_span = x_logs.max() - x_logs.min()
    points = []
    for x_log, y_val in zip(x_logs, y_vals):
        x = left + (x_log - x_logs.min()) / x_span * plot_w
        y = top + plot_h - (y_val - y_min) / (y_max - y_min + 1e-12) * plot_h
        points.append((x, y))
    draw.line(points, fill=color + (255,), width=4)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color + (255,))


def main():
    summary = load_summary()
    ap = summary[summary["method"] == "AP-FedAdam"].sort_values("l1_lambda")
    no_prox = summary[summary["method"] == "No-prox"].sort_values("l1_lambda")
    x_vals = ap["l1_lambda"].to_numpy(dtype=float)
    x_logs = np.log10(x_vals)
    x_labels = [f"{value:g}" for value in x_vals]

    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(32, True)
    label_font = font(22, True)
    small_font = font(18)
    note_font = font(19, True)
    blue = (31, 119, 180)
    red = (214, 39, 40)
    green = (44, 160, 44)
    black = (20, 20, 20)

    draw.text((55, 30), "Sparsity Stress Test Across L1 Penalties", fill=black, font=title_font)

    left, top, plot_w, plot_h = 105, 145, 560, 385
    mae_vals = np.r_[ap["final_mae"].to_numpy(float), no_prox["final_mae"].to_numpy(float)]
    y_min, y_max = 0.0, float(mae_vals.max() * 1.18)
    draw.text((left, top - 44), "A. Coefficient MAE", fill=black, font=label_font)
    draw.text((left - 4, top - 76), "MAE", fill=black, font=label_font)
    draw_axes(draw, left, top, plot_w, plot_h, x_logs, x_labels, y_min, y_max, lambda x: f"{x:.2f}")
    draw_line(draw, x_logs, ap["final_mae"].to_numpy(float), left, top, plot_w, plot_h, y_min, y_max, blue)
    draw_line(draw, x_logs, no_prox["final_mae"].to_numpy(float), left, top, plot_w, plot_h, y_min, y_max, red)
    legend_x, legend_y = left + plot_w - 148, top + 28
    for color, name in [(blue, "AP-FedAdam"), (red, "No-prox")]:
        draw.line([(legend_x, legend_y + 10), (legend_x + 32, legend_y + 10)], fill=color + (255,), width=5)
        draw.text((legend_x + 43, legend_y), name, fill=black, font=small_font)
        legend_y += 34
    best = ap.loc[ap["final_mae"].idxmin()]
    draw.text(
        (left + 22, top + plot_h + 62),
        f"Best AP-FedAdam: lambda={best['l1_lambda']:.0e}, MAE={best['final_mae']:.3f}",
        fill=blue,
        font=note_font,
    )
    draw.text((left + 155, top + plot_h + 102), "L1 penalty (lambda)", fill=black, font=label_font)

    left, top, plot_w, plot_h = 810, 145, 455, 385
    exact = ap["pred_zero_frac_1e8"].to_numpy(float)
    near = ap["pred_small_frac_1e4"].to_numpy(float)
    y_min, y_max = 0.0, max(0.05, float(np.r_[exact, near].max() * 1.20))
    draw.text((left, top - 44), "B. AP-FedAdam Sparsity", fill=black, font=label_font)
    draw.text((left - 4, top - 76), "Fraction", fill=black, font=label_font)
    draw_axes(draw, left, top, plot_w, plot_h, x_logs, x_labels, y_min, y_max, lambda x: f"{x:.2f}")
    draw_line(draw, x_logs, exact, left, top, plot_w, plot_h, y_min, y_max, blue)
    draw_line(draw, x_logs, near, left, top, plot_w, plot_h, y_min, y_max, green)
    legend_x, legend_y = left + plot_w - 145, top + 28
    for color, name in [(blue, "Exact zero"), (green, "Near zero")]:
        draw.line([(legend_x, legend_y + 10), (legend_x + 32, legend_y + 10)], fill=color + (255,), width=5)
        draw.text((legend_x + 43, legend_y), name, fill=black, font=small_font)
        legend_y += 34
    draw.text((left + 95, top + plot_h + 102), "L1 penalty (lambda)", fill=black, font=label_font)

    out_path = FIG_DIR / "fig3d_sparsity_stress_lambda.png"
    image.save(out_path)
    print(summary[[
        "l1_lambda",
        "method",
        "final_mae",
        "final_pearson",
        "pred_zero_frac_1e8",
        "pred_small_frac_1e4",
        "beta_l1_norm",
    ]].to_string(index=False))
    print(out_path)


if __name__ == "__main__":
    main()
