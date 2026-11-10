'Draw an editable conceptual workflow; no experimental values are plotted.'
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/distrisvg-mpl-2026-10-12")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-10-12"
C = dict(ink="#253039", muted="#58646D", line="#B9C3C8", blue="#316D91",
         teal="#26766D", light_blue="#EEF4F8", light_teal="#EFF7F4", gray="#F4F6F7")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def main(output_dir=OUT):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183/25.4, 148/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    texts = []

    def text(x, y, label, size=8, weight="normal", color=None, ha="left", **kw):
        artist = ax.text(x, y, label, fontsize=size, fontweight=weight, color=color or C["ink"],
                         ha=ha, va="center", linespacing=1.3, **kw)
        texts.append(artist)
        return artist

    def rect(x, y, w, h, fill, edge=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge or C["line"], lw=.65))

    def arrow(start, end, color=None, style="-|>", curve=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                    connectionstyle=f"arc3,rad={curve}", linewidth=.9,
                                    color=color or C["muted"], shrinkA=0, shrinkB=0))

    def matrix(x, y, w, h, colors, shift):
        for i in range(5):
            for j in range(4):
                value = colors[(i * 3 + j + shift) % len(colors)]
                ax.add_patch(Rectangle((x+j*w/4, y+i*h/5), w/4, h/5,
                                       facecolor=value, edgecolor="white", lw=.5))

    text(.03, .963, "a", 11, "bold")
    text(.068, .963, "Section-local inputs", 10, "bold")
    for i, (x, label) in enumerate(zip([.03, .365, .70], ["Site 1", "Site 2", "Site M"])):
        rect(x, .774, .27, .148, "white")
        text(x+.014, .900, label, 8, "bold")
        matrix(x+.023, .820, .052, .052, ["#DCE8EF", "#97B7CC", "#39789E", "#F3F6F7"], i)
        matrix(x+.113, .820, .045, .052, ["#E0EEE7", "#93B8A5", "#387E69", "#F3F6F7"], i+2)
        coords = [(a, b) for b in range(5) for a in range(5)
                  if ((a-2)/2.3)**2+((b-2)/2.5)**2 < 1]
        for a, b in coords:
            ax.plot(x+.204+a*.008, .823+b*.011, "o", ms=1.5, color="#89969E")
        text(x+.049, .795, r"Counts $Y_m$", 6.7, ha="center")
        text(x+.137, .795, r"Weights $\omega_m$", 6.7, ha="center")
        text(x+.223, .795, r"Positions $s_m$", 6.7, ha="center")
    text(.5, .747, "Counts and cell-type proportions remain at sites during fitting", 7.7, ha="center")

    rect(.085, .646, .83, .068, C["gray"])
    text(.5, .691, r"Common spatial representation $h(s)$", 8.5, "bold", ha="center")
    text(.5, .669, "Linear  |  Gaussian 1 and 2  |  Cosine 1 and 2", 7.5, ha="center")
    arrow((.5, .732), (.5, .717))
    text(.5, .622, "Defined before fitting from combined public-data coordinates", 7, ha="center", color=C["muted"])
    ax.plot([.5, .5, .265, .265], [.607, .586, .586, .568], color=C["line"], lw=.9)
    ax.plot([.5, .755, .755], [.586, .586, .568], color=C["line"], lw=.9)
    text(.5, .568, "Two analyses of the same shared model", 7, ha="center", color=C["muted"])

    text(.03, .530, "b", 11, "bold", color=C["blue"])
    text(.068, .530, "Sparse estimation", 9.5, "bold", color=C["blue"])
    text(.52, .530, "c", 11, "bold", color=C["teal"])
    text(.558, .530, "Gene-cell-type inference", 9.5, "bold", color=C["teal"])
    ax.plot([.5, .5], [.052, .515], color="#DCE2E5", lw=.6)

    rect(.07, .421, .39, .076, C["light_blue"])
    text(.265, .474, "At each site", 8, "bold", ha="center")
    text(.265, .446, "Adam-type update + soft threshold", 7.7, ha="center")
    arrow((.14, .421), (.14, .350), C["blue"])
    arrow((.39, .350), (.39, .421), C["blue"])
    text(.14, .385, "Parameter\nincrements", 6.7, ha="center", backgroundcolor="white")
    text(.39, .385, "Shared\nparameters", 6.7, ha="center", backgroundcolor="white")
    rect(.07, .257, .39, .093, C["light_blue"])
    text(.265, .328, "Coordinator", 8, "bold", ha="center")
    text(.265, .300, "Aggregate available cell-type blocks", 7.5, ha="center")
    text(.265, .276, "AP-FedAdam", 8, "bold", ha="center", color=C["blue"])
    arrow((.265, .257), (.265, .203), C["blue"])
    text(.265, .180, r"Shared sparse spatial effects $\hat{\beta}$", 8.3, "bold", ha="center", color=C["blue"])
    text(.265, .148, "Estimated separately within each basis family", 7, ha="center")
    text(.265, .105, "Each cell-type block aggregates only\nsites with the corresponding input", 7.2, ha="center")

    rect(.56, .421, .39, .076, C["light_teal"])
    text(.755, .474, "At each site", 8, "bold", ha="center")
    text(.755, .446, "Likelihood, gradient and full Hessian", 7.7, ha="center")
    arrow((.63, .421), (.63, .350), C["teal"])
    arrow((.88, .350), (.88, .421), C["teal"])
    text(.63, .385, "Local\nsummaries", 6.7, ha="center", backgroundcolor="white")
    text(.88, .385, "Common\nparameters", 6.7, ha="center", backgroundcolor="white")
    rect(.56, .257, .39, .093, C["light_teal"])
    text(.755, .328, "Independent all-variable refit", 8, "bold", ha="center")
    text(.755, .300, "Coefficients + dispersion", 7.6, ha="center")
    text(.755, .276, "Full nuisance-adjusted covariance", 7.4, ha="center")
    arrow((.755, .257), (.755, .221), C["teal"])
    text(.755, .205, "Joint Wald test for each gene-cell-type pair", 7.5, ha="center")
    text(.755, .178, r"$H_{0,gk}:\;\beta_{gk1}=\beta_{gk2}=0$", 8, ha="center")
    arrow((.755, .160), (.755, .137), C["teal"])
    text(.755, .120, "Cauchy combination across five families", 7.4, ha="center")
    text(.755, .097, "One BH adjustment across all pairs", 7.4, ha="center")
    text(.755, .067, "Candidate gene-cell-type pairs", 8.3, "bold", ha="center", color=C["teal"])
    text(.5, .025, "Schematic workflow. The refit uses its own initialization and all prespecified coefficients.",
         6.7, ha="center", color=C["muted"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    overflow = []
    for item in texts:
        b = item.get_window_extent(renderer)
        if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
            overflow.append(item.get_text())
    if overflow:
        raise RuntimeError(f"Text beyond canvas: {overflow}")
    overlaps = []
    boxes = [item.get_window_extent(renderer) for item in texts]
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                overlaps.append([texts[i].get_text(), texts[j].get_text()])
    if overlaps:
        raise RuntimeError(f"Overlapping text: {overlaps}")
    stem = output_dir / "distrisvg_workflow"
    for ext in ["pdf", "svg", "png", "tiff"]:
        fig.savefig(stem.with_suffix("."+ext), dpi=600 if ext=="tiff" else 250,
                    facecolor="white", metadata={"Title": "DistriSVG workflow"} if ext=="pdf" else None)
    (output_dir / "workflow_manifest.json").write_text(json.dumps({
        "kind": "conceptual schematic; no experimental values",
        "size_mm": [183, 148], "backend": "matplotlib",
        "parallel_branches": ["AP-FedAdam sparse estimation", "independent all-variable refit"],
        "input_contract": "counts, cell-type weights, coordinates; common basis fixed before fitting",
        "scope": "refit calibration assessed under complete composition inputs; no general FDR guarantee",
        "text_canvas_check": "passed", "text_overlap_check": "passed",
        "script": Path(__file__).name,
    }, indent=2)+"\n")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    main(parser.parse_args().output_dir)
