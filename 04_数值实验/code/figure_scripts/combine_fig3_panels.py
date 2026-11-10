#!/usr/bin/env python3
"""Combine the four Figure 3 panels into one manuscript-ready image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "overleaf" / "figs"

PANELS = [
    ("A", "fig3a_loss_convergence.png", "Training loss"),
    ("B", "fig3b_gradnorm_convergence.png", "Update norm"),
    ("C", "fig3c_ablation_mae_bar.png", "Ablation MAE"),
    ("D", "fig3d_sparsity_stress_lambda.png", "Sparsity stress"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_panel(path: Path, box: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    x = (box[0] - img.width) // 2
    y = (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def main() -> None:
    panel_w = 820
    panel_h = 520
    gutter = 52
    margin_x = 46
    margin_y = 58
    label_h = 34
    width = margin_x * 2 + panel_w * 2 + gutter
    height = margin_y * 2 + (panel_h + label_h) * 2 + gutter
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f_label = font(32, bold=True)
    f_title = font(20, bold=True)

    for idx, (letter, file_name, title) in enumerate(PANELS):
        row = idx // 2
        col = idx % 2
        x = margin_x + col * (panel_w + gutter)
        y = margin_y + row * (panel_h + label_h + gutter)
        draw.text((x, y - 38), letter, fill=(20, 20, 20), font=f_label)
        draw.text((x + 44, y - 31), title, fill=(55, 55, 55), font=f_title)
        panel = fit_panel(FIG_DIR / file_name, (panel_w, panel_h))
        canvas.paste(panel, (x, y))

    out_path = FIG_DIR / "fig3_ablation_combined.png"
    canvas.save(out_path, dpi=(300, 300))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
