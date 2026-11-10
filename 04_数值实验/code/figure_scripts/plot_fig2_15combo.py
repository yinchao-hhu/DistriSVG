from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "code" / "new_test1-5"
OUT_DIR = ROOT / "overleaf" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_MODES = ["linear", "gau", "cos"]
BASIS_MODES = ["linear", "gau1", "gau2", "cos1", "cos2"]
PATTERN = re.compile(
    r"DataMode_(?P<data>linear|gau|cos)_BasisMode_(?P<basis>linear|gau1|gau2|cos1|cos2)_Inference_FIXED\.csv$"
)


def load_summary():
    rows = []
    files = sorted(DATA_DIR.glob("CTSV_Fed_8_G_1000_L1_2e-07_Round1000_h_DataMode_*_BasisMode_*_Inference_FIXED.csv"))
    for path in files:
        match = PATTERN.search(path.name)
        if not match:
            continue
        data_mode = match.group("data")
        basis_mode = match.group("basis")
        df = pd.read_csv(path)
        abs_diff = (df["Learned_Beta"] - df["True_Beta"]).abs()
        true_nonzero = df["True_Beta"].abs() > 1e-12
        z_hit = df["Z_Score"].abs() > 1.96
        sensitivity_z196 = float(z_hit[true_nonzero].mean())
        type_i_z196 = float(z_hit[~true_nonzero].mean())
        rows.append(
            {
                "data_mode": data_mode,
                "basis_mode": basis_mode,
                "mae": float(abs_diff.mean()),
                "rmse": float(np.sqrt(((df["Learned_Beta"] - df["True_Beta"]) ** 2).mean())),
                "pearson": float(df[["True_Beta", "Learned_Beta"]].corr().iloc[0, 1]),
                "sensitivity_z196": sensitivity_z196,
                "type_i_z196": type_i_z196,
                "z_rate_1_96": float((df["Z_Score"].abs() > 1.96).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["data_mode"] = pd.Categorical(summary["data_mode"], DATA_MODES, ordered=True)
    summary["basis_mode"] = pd.Categorical(summary["basis_mode"], BASIS_MODES, ordered=True)
    return summary.sort_values(["data_mode", "basis_mode"])


def read_combo(data_mode, basis_mode):
    path = DATA_DIR / (
        "CTSV_Fed_8_G_1000_L1_2e-07_Round1000_h_"
        f"DataMode_{data_mode}_BasisMode_{basis_mode}_Inference_FIXED.csv"
    )
    return pd.read_csv(path)


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


def color_for(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        r = int(49 + u * (255 - 49))
        g = int(130 + u * (255 - 130))
        b = int(189 + u * (255 - 189))
    else:
        u = (t - 0.5) / 0.5
        r = int(255 + u * (202 - 255))
        g = int(255 + u * (0 - 255))
        b = int(255 + u * (32 - 255))
    return (r, g, b)


def draw_heatmap(summary):
    cell_w, cell_h = 180, 105
    left, top = 210, 150
    width = left + cell_w * len(BASIS_MODES) + 180
    height = top + cell_h * len(DATA_MODES) + 150
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = font(34, bold=True)
    label_font = font(23, bold=True)
    small_font = font(20)
    value_font = font(24, bold=True)

    draw.text((40, 35), "A. Beta Recovery Across Data and Basis Modes", fill=(20, 20, 20), font=title_font)
    draw.text((40, 82), "Cell value: mean absolute error of learned spatial coefficients", fill=(70, 70, 70), font=small_font)

    matrix = summary.pivot(index="data_mode", columns="basis_mode", values="mae")
    vmin, vmax = float(matrix.min().min()), float(matrix.max().max())

    for j, basis in enumerate(BASIS_MODES):
        x = left + j * cell_w
        draw.text((x + 20, top - 42), basis, fill=(20, 20, 20), font=label_font)

    for i, data_mode in enumerate(DATA_MODES):
        y = top + i * cell_h
        draw.text((45, y + 34), data_mode, fill=(20, 20, 20), font=label_font)
        for j, basis in enumerate(BASIS_MODES):
            x = left + j * cell_w
            val = float(matrix.loc[data_mode, basis])
            fill = color_for(val, vmin, vmax)
            draw.rectangle([x, y, x + cell_w - 6, y + cell_h - 6], fill=fill, outline=(255, 255, 255), width=3)
            text = f"{val:.3f}"
            bbox = draw.textbbox((0, 0), text, font=value_font)
            tx = x + (cell_w - (bbox[2] - bbox[0])) / 2 - 3
            ty = y + (cell_h - (bbox[3] - bbox[1])) / 2 - 6
            luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
            text_fill = (20, 20, 20) if luminance > 150 else (255, 255, 255)
            draw.text((tx, ty), text, fill=text_fill, font=value_font)

    cb_x = left + cell_w * len(BASIS_MODES) + 40
    cb_y = top
    cb_h = cell_h * len(DATA_MODES) - 6
    for k in range(cb_h):
        val = vmin + (vmax - vmin) * k / max(1, cb_h - 1)
        draw.line([(cb_x, cb_y + cb_h - k), (cb_x + 32, cb_y + cb_h - k)], fill=color_for(val, vmin, vmax))
    draw.rectangle([cb_x, cb_y, cb_x + 32, cb_y + cb_h], outline=(40, 40, 40), width=1)
    draw.text((cb_x + 45, cb_y - 5), f"{vmax:.2f}", fill=(20, 20, 20), font=small_font)
    draw.text((cb_x + 45, cb_y + cb_h - 20), f"{vmin:.2f}", fill=(20, 20, 20), font=small_font)
    draw.text((left + 280, height - 85), "Basis mode used for fitting", fill=(40, 40, 40), font=label_font)
    draw.text((20, top + 325), "Data mode", fill=(40, 40, 40), font=label_font)

    img.save(OUT_DIR / "fig2a_15combo_mae_heatmap.png")


def draw_scatter(data_mode="cos", basis_mode="cos2"):
    df = read_combo(data_mode, basis_mode)
    x = df["True_Beta"].to_numpy()
    y = df["Learned_Beta"].to_numpy()
    nonzero = np.abs(x) > 1e-12
    mae = float(np.mean(np.abs(y - x)))
    corr = float(np.corrcoef(x, y)[0, 1])

    w, h = 1220, 780
    left, top, plot_w, plot_h = 180, 110, 730, 560
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(34, bold=True)
    label_font = font(24, bold=True)
    small_font = font(21)

    draw.text((40, 35), f"B. True vs Learned Beta ({data_mode} data, {basis_mode} basis)", fill=(20, 20, 20), font=title_font)
    draw.text((40, 75), f"MAE = {mae:.3f}; Pearson r = {corr:.3f}", fill=(70, 70, 70), font=small_font)

    lim = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 5.0))
    lim = np.ceil(lim)

    def px(val):
        return left + (val + lim) / (2 * lim) * plot_w

    def py(val):
        return top + plot_h - (val + lim) / (2 * lim) * plot_h

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(30, 30, 30), width=2)
    for tick in np.linspace(-lim, lim, 5):
        xx = px(tick)
        yy = py(tick)
        draw.line([(xx, top), (xx, top + plot_h)], fill=(220, 220, 220, 255), width=1)
        draw.line([(left, yy), (left + plot_w, yy)], fill=(220, 220, 220, 255), width=1)
        draw.text((xx - 18, top + plot_h + 10), f"{tick:.0f}", fill=(50, 50, 50), font=small_font)
        draw.text((left - 58, yy - 10), f"{tick:.0f}", fill=(50, 50, 50), font=small_font)

    draw.line([(px(-lim), py(-lim)), (px(lim), py(lim))], fill=(40, 40, 40, 255), width=2)

    rng = np.random.default_rng(1)
    order = np.r_[np.where(~nonzero)[0], np.where(nonzero)[0]]
    sample = order
    if len(order) > 10000:
        sample = rng.choice(order, size=10000, replace=False)
    for idx in sample:
        color = (49, 130, 189, 58) if not nonzero[idx] else (202, 0, 32, 145)
        r = 2 if not nonzero[idx] else 3
        xx = px(float(x[idx]))
        yy = py(float(y[idx]))
        draw.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color)

    draw.text((left + plot_w / 2 - 70, top + plot_h + 55), "True beta", fill=(20, 20, 20), font=label_font)
    draw.text((28, top + plot_h / 2 - 20), "Learned", fill=(20, 20, 20), font=label_font)
    draw.text((28, top + plot_h / 2 + 10), "beta", fill=(20, 20, 20), font=label_font)

    lx, ly = left + plot_w + 55, top + 85
    draw.ellipse([lx, ly, lx + 16, ly + 16], fill=(202, 0, 32, 180))
    draw.text((lx + 28, ly - 4), "true nonzero", fill=(20, 20, 20), font=small_font)
    draw.ellipse([lx, ly + 42, lx + 16, ly + 58], fill=(49, 130, 189, 100))
    draw.text((lx + 28, ly + 38), "true zero", fill=(20, 20, 20), font=small_font)

    img.save(OUT_DIR / "fig2b_true_vs_learned_scatter.png")


def draw_celltype_mae(summary):
    combos = [("linear", "linear"), ("gau", "gau1"), ("gau", "gau2"), ("cos", "cos2")]
    rows = []
    for data_mode, basis_mode in combos:
        df = read_combo(data_mode, basis_mode)
        df["abs_diff"] = (df["Learned_Beta"] - df["True_Beta"]).abs()
        for ct, sub in df.groupby("Cell_Type"):
            rows.append(
                {
                    "combo": f"{data_mode}/{basis_mode}",
                    "cell_type": int(ct),
                    "mae": float(sub["abs_diff"].mean()),
                }
            )
    ct_summary = pd.DataFrame(rows)
    ct_summary.to_csv(OUT_DIR / "fig2_celltype_mae_summary.csv", index=False)
    grouped = ct_summary.groupby("cell_type")["mae"].agg(["mean", "std"]).reset_index()

    w, h = 950, 680
    left, top, plot_w, plot_h = 115, 120, 720, 430
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(34, bold=True)
    label_font = font(24, bold=True)
    small_font = font(21)
    value_font = font(20, bold=True)

    draw.text((40, 35), "C. Cell-Type-Stratified Beta Recovery", fill=(20, 20, 20), font=title_font)
    draw.text((40, 75), "Mean absolute error across correctly specified or near-correct basis settings", fill=(70, 70, 70), font=small_font)

    ymax = float(np.ceil((grouped["mean"] + grouped["std"].fillna(0)).max() * 10) / 10)
    ymax = max(ymax, 0.3)
    for tick in np.linspace(0, ymax, 4):
        yy = top + plot_h - tick / ymax * plot_h
        draw.line([(left, yy), (left + plot_w, yy)], fill=(220, 220, 220, 255), width=1)
        draw.text((left - 70, yy - 10), f"{tick:.2f}", fill=(50, 50, 50), font=small_font)
    draw.line([(left, top), (left, top + plot_h)], fill=(30, 30, 30), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(30, 30, 30), width=2)

    bar_gap = 25
    n = len(grouped)
    bar_w = (plot_w - bar_gap * (n + 1)) / n
    for _, row in grouped.iterrows():
        i = int(row["cell_type"])
        mean = float(row["mean"])
        std = float(row["std"]) if np.isfinite(row["std"]) else 0.0
        x0 = left + bar_gap + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        y0 = top + plot_h - mean / ymax * plot_h
        draw.rectangle([x0, y0, x1, top + plot_h], fill=(49, 130, 189, 230))
        yerr0 = top + plot_h - min(ymax, mean + std) / ymax * plot_h
        yerr1 = top + plot_h - max(0, mean - std) / ymax * plot_h
        cx = (x0 + x1) / 2
        draw.line([(cx, yerr0), (cx, yerr1)], fill=(20, 20, 20), width=2)
        draw.line([(cx - 12, yerr0), (cx + 12, yerr0)], fill=(20, 20, 20), width=2)
        draw.line([(cx - 12, yerr1), (cx + 12, yerr1)], fill=(20, 20, 20), width=2)
        draw.text((cx - 12, top + plot_h + 12), str(i), fill=(20, 20, 20), font=small_font)
        draw.text((cx - 30, y0 - 30), f"{mean:.2f}", fill=(20, 20, 20), font=value_font)

    draw.text((left + plot_w / 2 - 70, top + plot_h + 55), "Cell type", fill=(20, 20, 20), font=label_font)
    draw.text((20, top + plot_h / 2 - 18), "MAE", fill=(20, 20, 20), font=label_font)
    img.save(OUT_DIR / "fig2c_celltype_mae_bar.png")


def main():
    summary = load_summary()
    summary.to_csv(OUT_DIR / "fig2_15combo_summary.csv", index=False)
    draw_heatmap(summary)
    draw_scatter()
    draw_celltype_mae(summary)
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'fig2_15combo_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'fig2a_15combo_mae_heatmap.png'}")
    print(f"Wrote {OUT_DIR / 'fig2b_true_vs_learned_scatter.png'}")
    print(f"Wrote {OUT_DIR / 'fig2c_celltype_mae_bar.png'}")


if __name__ == "__main__":
    main()
