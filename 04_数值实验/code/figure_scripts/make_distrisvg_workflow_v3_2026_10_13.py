"""Modular DistriSVG workflow with conceptual glyphs and archived validation.

Panels a-c are original schematic artwork, not empirical tissue reconstructions.
Panel d reads existing complete-input calibration records; no fitting is run.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/distrisvg-mpl-2026-10-13")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-10-13-v3"
RESULTS = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25"
C = {"ink": "#242C31", "muted": "#56646C", "edge": "#B8C5CC",
     "blue": "#387A9F", "blue_light": "#E7F0F6", "teal": "#317E71",
     "teal_light": "#E6F1ED", "gold": "#AB833A", "gold_light": "#F4EEDC",
     "coral": "#CC806B", "gray": "#F0F3F4"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 6.2, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def read_validation(metrics_path, summary_path, output_dir):
    with metrics_path.open() as f:
        records = list(csv.DictReader(f))
    with summary_path.open() as f:
        summaries = list(csv.DictReader(f))
    chosen = lambda r: all(r[k] == v for k, v in {
        "signal": "sparse", "policy": "complete", "covariance": "model",
        "adjustment": "acat_bh"}.items())
    rows = [r for r in records if chosen(r)]
    stats = {r["metric"]: r for r in summaries if chosen(r) and r["metric"] in {"power", "fdp"}}
    assert len(rows) == 50 and len({r["repeat"] for r in rows}) == 50
    assert set(stats) == {"power", "fdp"}
    for metric, row in stats.items():
        assert abs(np.mean([float(r[metric]) for r in rows])-float(row["mean"])) < 1e-12
        assert int(row["n_repeats"]) == 50
    data = output_dir / "source_data"
    data.mkdir(exist_ok=True)
    for name, values in [("complete_input_replicates.csv", rows),
                          ("complete_input_summary.csv", list(stats.values()))]:
        with (data / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return stats, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [metrics_path, summary_path]}


def main(output_dir, metrics_path, summary_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, provenance = read_validation(metrics_path, summary_path, output_dir)
    W, H = 183, 150
    fig = plt.figure(figsize=(W/25.4, H/25.4), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
    ax.axis("off")
    texts, owned = [], []
    current = None
    bounds = {}

    def text(x, y, value, size=6.2, weight="normal", ha="left", color=None, **kw):
        t = ax.text(x, y, value, fontsize=max(size, 5.8), fontweight=weight,
                    ha=ha, va="center", linespacing=1.2, color=color or C["ink"], **kw)
        texts.append(t)
        owned.append(current)
        return t

    def line(xs, ys, color=None, lw=.6, **kw):
        ax.plot(xs, ys, color=color or C["edge"], lw=lw, **kw)

    def rect(x, y, w, h, fill="white", edge=None, lw=.6, **kw):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                               edgecolor=edge or "none", lw=lw, **kw))

    def arrow(a, b, color=None, curve=0, lw=.8):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                    connectionstyle=f"arc3,rad={curve}",
                                    lw=lw, color=color or C["muted"], shrinkA=0, shrinkB=0))

    def module(letter, title, x, y, w, h, fill, edge):
        nonlocal current
        current = letter
        bounds[letter] = (x, y, x+w, y+h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc="white", ec=edge, lw=.8))
        rect(x+.4, y+h-8.5, w-.8, 8.1, fill)
        text(x+2, y+h-4.5, letter, 8, "bold")
        text(x+7, y+h-4.5, title, 7, "bold")

    def matrix(x, y, w, h, data, palette):
        data = np.asarray(data)
        nr, nc = data.shape
        for r in range(nr):
            for c in range(nc):
                rect(x+c*w/nc, y+(nr-r-1)*h/nr, w/nc, h/nr,
                     palette[int(data[r, c])], "white", .25)
        rect(x, y, w, h, "none", C["edge"], .35)

    def coordinates(x, y, w, h, phase):
        line([x, x, x+w], [y+h, y, y], C["edge"], .4)
        for i in range(4):
            for j in range(4):
                if (i+j+phase) % 7 != 0:
                    ax.add_patch(Circle((x+1.5+i*w/4.9, y+1+j*h/4.8), .38,
                                        fc=C["muted"], ec="none"))

    def server(x, y):
        for k in range(3):
            rect(x, y+k*1.6, 4, 1.3, "#E4EBEF", C["muted"], .35)
            ax.add_patch(Circle((x+.7, y+k*1.6+.65), .16, fc=C["blue"]))

    module("a", "Client-held inputs", 2, 38, 43, 110, C["blue_light"], C["blue"])
    text(23.5, 134.2, "Separate data sources", 6.2, "bold", ha="center")
    for k, (base, label) in enumerate([(106, "Client 1"), (78, "Client 2"), (50, "Client M")]):
        rect(5, base, 37, 23, "#FAFCFD", C["edge"], .55, linestyle=(0, (3, 2)))
        text(7, base+19.4, label, 6.3, "bold")
        server(35, base+17)
        counts = (np.arange(24).reshape(6, 4)+np.arange(6)[:, None]*2+k) % 4
        matrix(7.5, base+6.7, 8.5, 8, counts, ["#E9F1F6", "#B8D1E0", "#79ACC6", C["blue"]])
        proportions = (np.arange(18).reshape(6, 3)+np.arange(6)[:, None]+k) % 3
        matrix(20, base+6.7, 7.5, 8, proportions, ["#DDEDE6", "#9FC6B6", "#558F7D"])
        coordinates(31.5, base+6.7, 8, 8, k)
        text(11.8, base+3.4, "Counts", 5.8, ha="center")
        text(23.8, base+3.4, "Weights", 5.8, ha="center")
        text(35.5, base+3.4, "Coords", 5.8, ha="center")
    text(23.5, 45, "Counts and weights stay local", 5.8, "bold", ha="center")
    text(23.5, 41.2, "Exchange model information only", 5.8, ha="center")

    module("b", "DistriSVG statistical framework", 51, 38, 75, 110, C["teal_light"], C["teal"])
    text(88.5, 134, "Shared negative-binomial spatial model", 6.3, "bold", ha="center")
    text(88.5, 127, r"$Y_{mig}\sim\mathrm{NB}(\mu_{mig},\psi_g)$", 6.8, ha="center")
    text(88.5, 120, r"$\log\mu_{mig}=\log a_{mi}+\sum_k\omega_{mik}\eta_{gk}(s_{mi})$", 6.4, ha="center")
    text(88.5, 112.5, r"$\eta_{gk}(s)=\theta_{gk}+\beta_{gk}^{\mathsf{T}}h(s)$", 6.5, ha="center")
    for x, family in zip([58, 71, 84, 97, 110], ["Linear", "G1", "G2", "C1", "C2"]):
        u = np.linspace(-1, 1, 70)
        f = {"Linear": .5+.42*u, "G1": np.exp(-u*u/.5), "G2": np.exp(-u*u/.15),
             "C1": .5+.45*np.cos(np.pi*u), "C2": .5+.45*np.cos(2*np.pi*u)}[family]
        line([x, x, x+9], [106.7, 100.7, 100.7], lw=.4)
        line(x+4.5*(u+1), 101+5*f, C["teal"], .85)
        text(x+4.5, 98, family, 5.8, ha="center")
    text(88.5, 92.9, "Five families fitted separately in a common frame", 5.8, ha="center")
    line([54, 123], [89, 89], C["edge"], .55)
    text(88.5, 86, "Independent estimation and inference", 6.1, "bold", ha="center")
    line([89, 89], [41, 82], C["edge"], .45)
    text(71, 79.3, "Sparse fitting", 6.3, "bold", ha="center")
    text(71, 74.5, "AP-FedAdam", 6.2, "bold", ha="center")
    line([56, 67], [68.5, 68.5], lw=.4)
    line([61.5, 61.5], [63.5, 72.5], lw=.4)
    line([56, 60, 63, 67], [64, 68.5, 68.5, 72], C["blue"], .95)
    text(77, 68.5, "Local\nadaptive\nsteps", 5.8, ha="center")
    text(71, 60.5, "Soft threshold", 5.8, ha="center")
    arrow((71, 57.8), (71, 55), C["blue"])
    text(71, 51.8, "Available-block\naggregation", 6, ha="center")
    arrow((55, 53), (55, 71), C["blue"], curve=-.25, lw=.6)
    matrix(59, 41.8, 10, 5.7, [[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0]],
           [C["gray"], C["blue"], C["coral"]])
    text(77.3, 44.6, "Shared\neffects", 5.8, ha="center")

    text(107, 79.3, "Inference", 6.3, "bold", ha="center")
    text(107, 74.3, "Independent all-variable refit", 5.8, ha="center")
    text(107, 69.2, "Coefficients + dispersion", 5.8, ha="center")
    text(107, 64, "Full nuisance covariance", 5.8, ha="center")
    line([92, 122], [60.4, 60.4], lw=.4)
    text(107, 57.2, "Joint Wald (2 d.f.)", 5.8, ha="center")
    text(107, 52, r"$H_0:\beta_{gk1}=\beta_{gk2}=0$", 5.8, ha="center")
    text(107, 46.9, "Cauchy across five families", 5.8, ha="center")
    text(107, 41.8, "One BH adjustment", 5.8, "bold", ha="center")

    module("c", "Integrated ctSVG output", 132, 38, 49, 110, C["gold_light"], C["gold"])
    text(156.5, 134.1, "What is a ctSVG?", 6.6, "bold", ha="center")
    text(148, 128.8, "Client 1", 5.8, ha="center")
    text(172, 128.8, "Client M", 5.8, ha="center")
    text(159.8, 128.8, "...", 5.8, ha="center")
    cmap = LinearSegmentedColormap.from_list("expression", ["#EEF5F9", "#80B1CC", "#225E85"])
    for typ, cy in [("A", 117.5), ("B", 98.7)]:
        text(136.2, cy, "Type "+typ, 5.8, ha="center", rotation=90)
        for client, cx in enumerate([148, 172]):
            rect(cx-7.5, cy-6.7, 15, 13.4, "#FAFBFC", "#D9E1E6", .35)
            for i in range(6):
                for j in range(6):
                    if (i, j) in ([(0, 0), (0, 5), (5, 0)] if client==0 else [(5, 5), (0, 0), (5, 0)]):
                        continue
                    value = .12+.85*i/5 if typ=="A" else .45
                    ax.add_patch(Circle((cx-5.8+2.3*i, cy-5.1+2.05*j), .79,
                                        fc=cmap(value), ec="white", lw=.25))
    text(156.5, 87.4, "Gene g varies spatially within A,\nnot within B", 6.1, ha="center")
    text(141, 79.8, "Low", 5.8, ha="center")
    for j in range(30):
        rect(147+j*.5, 78.8, .5, 2, cmap(j/29))
    text(172, 79.8, "High", 5.8, ha="center")
    line([135, 178], [74.9, 74.9], C["edge"], .5)
    text(156.5, 71.1, "Gene-cell-type candidates", 6.4, "bold", ha="center")
    text(161.3, 66.6, "Cell types", 5.8, ha="center")
    for cx, label in [(151.3, "A"), (161.3, "B"), (171.3, "C")]:
        text(cx, 62.7, label, 5.8, ha="center")
    matrix(146.3, 46.6, 30, 13.8, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["#F2F3EF", C["teal"]])
    for cy, label in [(58.1, "g"), (53.5, r"$g_2$"), (48.9, r"$g_3$")]:
        text(142.4, cy, label, 5.8, ha="center")
    text(156.5, 41.8, "Combined p values and BH q values", 5.8, ha="center")

    current = None
    for y in [118, 90, 62]:
        arrow((45.5, y), (50.5, y), C["blue"])
        arrow((50.5, y-5), (45.5, y-5), C["blue"])
    line([119, 129, 129], [41.8, 41.8, 53.5], C["teal"], .8)
    arrow((129, 53.5), (131.4, 53.5), C["teal"])

    module("d", "Existing complete-input simulation", 2, 2, 179, 31,
           "#EFF2F3", "#A6B4BB")
    text(7, 20.3, "50 sparse-signal response replicates", 6, "bold")
    text(7, 14.9, "Information covariance + Cauchy/BH", 6)
    text(7, 8.7, "Points: means; bars: 95% intervals", 5.8, color=C["muted"])
    for metric, x0, x1, xmax, label, col in [
            ("power", 85, 121, 1., "Power", C["blue"]),
            ("fdp", 139, 175, .1, "Mean FDP", C["teal"])]:
        r = stats[metric]
        mean, lo, hi = [float(r[k]) for k in ["mean", "lower", "upper"]]
        text((x0+x1)/2, 20.4, f"{label}  {mean:.3f}", 6.5, "bold", ha="center")
        scale = lambda v: x0+(x1-x0)*v/xmax
        line([x0, x1], [9.5, 9.5], C["edge"], .5)
        for value in [0, xmax/2, xmax]:
            line([scale(value), scale(value)], [9.5, 8.6], C["edge"], .5)
            text(scale(value), 6, f"{value:g}", 5.8, ha="center")
        line([scale(lo), scale(hi)], [14, 14], col, 1)
        for v in [lo, hi]:
            line([scale(v), scale(v)], [12.7, 15.3], col, .8)
        ax.add_patch(Circle((scale(mean), 14), .85, fc=col, ec="white", lw=.35))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    overflow, overlaps, module_overflow = [], [], []
    for i, b in enumerate(boxes):
        if b.x0<fig.bbox.x0 or b.x1>fig.bbox.x1 or b.y0<fig.bbox.y0 or b.y1>fig.bbox.y1:
            overflow.append(texts[i].get_text())
        if owned[i] is not None:
            x0, y0, x1, y1 = bounds[owned[i]]
            p0, p1 = ax.transData.transform([(x0, y0), (x1, y1)])
            if b.x0<p0[0] or b.x1>p1[0] or b.y0<p0[1] or b.y1>p1[1]:
                module_overflow.append(texts[i].get_text())
        for j in range(i+1, len(boxes)):
            c = boxes[j]
            if min(b.x1,c.x1)-max(b.x0,c.x0)>1 and min(b.y1,c.y1)-max(b.y0,c.y0)>1:
                overlaps.append([texts[i].get_text(), texts[j].get_text()])
    if overflow or overlaps or module_overflow:
        raise RuntimeError(json.dumps({"overflow": overflow, "overlaps": overlaps,
                                       "module_overflow": module_overflow}, indent=2))
    stem = output_dir / "distrisvg_workflow_v3"
    for ext in ["pdf", "svg", "png", "tiff"]:
        fig.savefig(stem.with_suffix("."+ext), dpi=600 if ext=="tiff" else 260,
                    facecolor="white", metadata={"Title": "DistriSVG modular workflow"} if ext=="pdf" else None)
    manifest = {"size_mm": [W, H], "backend": "Python/Matplotlib", "body_font_pt": [5.8, 7],
                "kind": "schematic modules a-c plus archived quantitative validation d",
                "conceptual_panels": ["a", "b", "c"], "data_panel": "d",
                "inputs_sha256": provenance, "validation": stats,
                "checks": {"text_overlap": "passed", "canvas_containment": "passed",
                           "module_text_containment": "passed"},
                "scope": "complete-input fixed-design calibration, not general superiority or FDR control",
                "script": Path(__file__).name}
    (output_dir / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "calibration_metrics.csv")
    parser.add_argument("--summary", type=Path, default=RESULTS / "calibration_summary.csv")
    args = parser.parse_args()
    main(args.output_dir, args.metrics, args.summary)
