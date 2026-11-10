"""Draw the DistriSVG conceptual workflow. No empirical values are plotted.

Three horizontal panels distinguish local inputs, sparse estimation and an
independently initialized all-variable refit. No sparse-to-refit arrow is used.
All scientific glyphs are schematic. Exports retain editable PDF/SVG text.
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/distrisvg-mpl-2026-10-12")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-10-12-v2"
C = dict(ink="#252B30", muted="#58636A", line="#C6CED2", blue="#3478A3",
         pale_blue="#F0F6FA", teal="#248579", pale_teal="#F0F8F5",
         coral="#CD7968", gold="#C9A457", gray="#EEF1F2")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 6.7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def main(output_dir=OUT):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183/25.4, 161/25.4), facecolor="white", dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 183), ylim=(0, 161), aspect="equal")
    ax.axis("off")
    texts = []

    def text(x, y, label, size=6.7, weight="normal", ha="left", color=None, **kw):
        item = ax.text(x, y, label, fontsize=max(size, 5.8), fontweight=weight, ha=ha,
                       va="center", linespacing=1.25, color=color or C["ink"], **kw)
        texts.append(item)
        return item

    def line(xs, ys, color=None, lw=.6, **kw):
        ax.plot(xs, ys, color=color or C["line"], lw=lw, **kw)

    def rect(x, y, w, h, fill="white", edge=None, lw=.65):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                               edgecolor=edge or "none", linewidth=lw))

    def arrow(start, end, color=None, curve=0, lw=.8):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    connectionstyle=f"arc3,rad={curve}", mutation_scale=7,
                                    linewidth=lw, color=color or C["muted"],
                                    shrinkA=0, shrinkB=0))

    def grid(x, y, w, h, data, palette):
        rows, cols = np.asarray(data).shape
        for i in range(rows):
            for j in range(cols):
                rect(x+j*w/cols, y+(rows-1-i)*h/rows, w/cols, h/rows,
                     palette[int(data[i][j])], "white", .35)
        rect(x, y, w, h, "none", C["line"], .35)

    def tissue(cx, cy, shift):
        theta = np.linspace(0, 2*np.pi, 120)
        radius = 1+.11*np.sin(3*theta+shift)+.065*np.cos(5*theta-shift)
        outline = np.c_[cx+9*radius*np.cos(theta), cy+6.3*radius*np.sin(theta)]
        ax.add_patch(Polygon(outline, fc="#F7F8F8", ec=C["line"], lw=.5))
        for row in range(-4, 5):
            for col in range(-6, 7):
                u = (col+.5*(row % 2))*1.28
                v = row*1.27
                if (u/8.1)**2+(v/5.6)**2 < .93:
                    region = (col+row+shift) % 6
                    fill = ["#A4C8DC", "#A4C8DC", "#A4C8DC", "#9BC7B5",
                            "#9BC7B5", "#E8BEAB"][region]
                    ax.add_patch(Circle((cx+u, cy+v), .48, fc=fill, ec="white", lw=.25))

    def basis(x, y, kind):
        u = np.linspace(-1, 1, 80)
        vals = {"Linear": .5+.4*u, "G1": np.exp(-u*u/.45),
                "G2": np.exp(-u*u/.13), "C1": .5+.45*np.cos(np.pi*u),
                "C2": .5+.45*np.cos(2*np.pi*u)}[kind]
        line([x, x, x+10], [y+6, y, y], lw=.45)
        line(x+5*(u+1), y+.5+5*vals, C["blue"], .9)
        text(x+5, y-2.6, kind, 5.7, ha="center")

    def panel(letter, title, y, accent):
        text(4, y, letter, 8, "bold")
        text(9, y, title, 7, "bold")
        line([9, 179], [y-4, y-4], accent, .7)

    panel("a", "Local tissue data and a shared spatial model", 155, C["line"])
    for i, (cx, name) in enumerate(zip([20, 47, 76], ["Site 1", "Site 2", "Site M"])):
        text(cx, 145.8, name, 6.6, "bold", ha="center")
        tissue(cx, 135.7, i)
        count = (np.arange(24).reshape(6, 4)*3+i+np.arange(6)[:, None]) % 5
        grid(cx-9, 119.3, 8, 7, count,
             ["#F1F5F7", "#D4E6EF", "#98BFD5", "#639CBF", C["blue"]])
        props = (np.arange(18).reshape(6, 3)+i) % 3
        grid(cx+1, 119.3, 8, 7, props, ["#D6EADD", "#A5CDB8", "#619E88"])
        text(cx-5, 116, r"$Y_m$", 6, ha="center")
        text(cx+5, 116, r"$\omega_m$", 6, ha="center")
    text(61.5, 136, "...", 7, ha="center")
    text(48, 110.5, "Counts + cell-type proportions + coordinates", 6.1, ha="center")
    text(48, 106, "Counts and proportions stay local during fitting", 5.8,
         ha="center", color=C["muted"])
    arrow((88, 133), (97, 133))
    text(139, 145.8, "Shared negative-binomial model", 6.8, "bold", ha="center")
    text(139, 138.8, "Cell-type-specific spatial effects", 6.2, ha="center")
    text(139, 132, r"$\eta_{gk}(s)=\theta_{gk}+\beta_{gk}^{\mathsf{T}}h(s)$", 7, ha="center")
    text(139, 125.5, "Five basis families, fitted separately", 6.2, ha="center")
    for x, family in zip([102, 117, 132, 147, 162], ["Linear", "G1", "G2", "C1", "C2"]):
        basis(x, 114, family)
    text(139, 105.7, "One common coordinate transformation before fitting", 5.7,
         ha="center", color=C["muted"])

    panel("b", "Sparse estimation with AP-FedAdam", 97, C["blue"])
    rect(9, 69, 54, 19, C["pale_blue"])
    rect(87, 69, 47, 19, C["pale_blue"])
    line([12, 24], [77, 77], C["line"], .45)
    line([18, 18], [71, 84], C["line"], .45)
    line([12, 16, 20, 24], [72, 77, 77, 82], C["blue"], 1.05)
    text(29, 82.3, "Local adaptive update", 6.5, "bold")
    text(29, 76.5, "Soft threshold", 6.4)
    text(36, 64.5, "Repeated at each site", 6.1, ha="center")
    arrow((64, 82), (86, 82), C["blue"])
    arrow((86, 73), (64, 73), C["blue"])
    text(75, 86.5, "Increments", 5.7, ha="center")
    text(75, 68.4, "Parameters", 5.7, ha="center")
    text(110.5, 82.5, "Coordinator", 6.5, "bold", ha="center")
    for i, fill in enumerate(["#86B4CF", "#A9CCDF", "#D0E3ED"]):
        rect(93+i*7, 72, 6, 4, fill)
    arrow((115, 74), (121, 74), C["blue"], lw=.65)
    rect(123, 72, 6, 4, C["blue"])
    text(110.5, 64.5, "Aggregate available cell-type blocks", 5.9, ha="center")
    arrow((136, 78), (147, 78), C["blue"])
    coeff = [[0, 0, 2, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
             [0, 2, 0, 1, 0], [0, 0, 0, 0, 2], [0, 0, 1, 0, 0]]
    grid(153, 70, 20, 17, coeff, ["#F2F4F5", C["blue"], C["coral"]])
    for x, fill, label in [(153, C["blue"], "-"), (160.5, "#E9EDEF", "0"),
                           (168, C["coral"], "+")]:
        rect(x, 89, 2.2, 1.8, fill)
        text(x+3.1, 89.8, label, 5.5)
    text(149.7, 78.5, "Genes", 5.5, ha="center", rotation=90)
    text(163, 66.5, "Cell types", 5.5, ha="center")
    text(163, 61.7, r"Sparse effects $\hat{\beta}$", 6.6, "bold", ha="center")

    panel("c", "Gene-cell-type inference from an independent refit", 54, C["teal"])
    text(36, 44.5, "Own initialization; all variables", 6.1, "bold", ha="center")
    rect(9, 19, 54, 21, C["pale_teal"])
    text(36, 35.4, "Distributed likelihood refit", 6.5, "bold", ha="center")
    text(36, 29.4, "Local likelihood, gradient and full Hessian", 5.7, ha="center")
    text(36, 23.5, "Joint coefficients and dispersion", 6, ha="center")
    text(36, 14.5, "Full nuisance-adjusted covariance", 5.9, ha="center")
    arrow((65, 29), (74, 29), C["teal"])
    text(94, 40, "Joint Wald tests", 6.5, "bold", ha="center")
    text(94, 32.7, r"$H_{0,gk}:\beta_{gk1}=\beta_{gk2}=0$", 6.5, ha="center")
    for i in range(5):
        rect(76+i*7.4, 21.3, 6.2, 6.3, C["pale_teal"], "#A5CCC2", .5)
        text(79.1+i*7.4, 24.4, r"$p^{("+str(i+1)+")}$", 5.8, ha="center")
    text(94, 16.7, "One p value per pair and family", 5.8, ha="center")
    line([114, 120, 120], [24.4, 24.4, 33.2], C["teal"], .8)
    arrow((120, 33.2), (136, 33.2), C["teal"])
    text(152, 40, "Combine across five families", 6.2, "bold", ha="center")
    text(152, 33.2, "Cauchy combination", 6.5, ha="center")
    arrow((152, 29.6), (152, 25.6), C["teal"])
    text(152, 22, "One BH adjustment across pairs", 6.1, ha="center")
    arrow((152, 18.8), (152, 14.8), C["teal"])
    text(152, 10.8, "Candidate gene-cell-type pairs", 6.6, "bold", ha="center")
    text(4, 4, "Conceptual illustration. Sparse estimation does not initialize or select variables for the refit.",
         5.5, color=C["muted"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    boxes = [item.get_window_extent(renderer) for item in texts]
    overflow = [texts[i].get_text() for i, b in enumerate(boxes)
                if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1]
    overlaps = []
    for i, a in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            b = boxes[j]
            if min(a.x1, b.x1)-max(a.x0, b.x0)>1 and min(a.y1, b.y1)-max(a.y0, b.y0)>1:
                overlaps.append([texts[i].get_text(), texts[j].get_text()])
    if overflow or overlaps:
        raise RuntimeError(json.dumps({"overflow": overflow, "overlaps": overlaps}, indent=2))
    stem = output_dir / "distrisvg_workflow_v2"
    for ext in ["pdf", "svg", "png", "tiff"]:
        fig.savefig(stem.with_suffix("."+ext), dpi=600 if ext=="tiff" else 240,
                    facecolor="white", metadata={"Title": "DistriSVG workflow"} if ext=="pdf" else None)
    manifest = {
        "kind": "conceptual schematic; no experimental values", "size_mm": [183, 161],
        "backend": "matplotlib", "font": "Arial", "body_font_pt": [5.8, 7],
        "panel_font_pt": 8, "text_canvas_check": "passed", "text_overlap_check": "passed",
        "panels": {"a": "local inputs and shared model", "b": "AP-FedAdam sparse estimation",
                   "c": "independently initialized all-variable refit and testing"},
        "scientific_contract": ["five families fitted separately", "Cauchy then one BH",
                                "no sparse-to-refit dependency", "nuisance-adjusted covariance",
                                "public-data coordinate preprocessing is described in the caption"],
        "glyphs": "Original illustrative tissue spots, matrices and basis curves; not data or anatomy",
        "script": Path(__file__).name,
    }
    (output_dir / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    main(parser.parse_args().output_dir)
