"""Graphical candidate for the DistriSVG overview, independent of manuscript v4.

All glyphs in a-c are conceptual. Panel d alone uses archived quantitative data.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-10-13-v5-candidate"
RESULTS = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25"
C = {"ink": "#27373C", "muted": "#62727A", "edge": "#B7C7CC",
     "blue": "#33749C", "blue_mid": "#83ADC9", "blue_pale": "#D4E5F0",
     "green": "#348778", "green_mid": "#88B7A7", "green_pale": "#D8E9DF",
     "gold": "#AC813D", "gold_pale": "#F1E1BB", "coral": "#C37664"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def validation(metrics, summary, out):
    selection = {"signal": "sparse", "policy": "complete", "covariance": "model", "adjustment": "acat_bh"}
    selected = lambda row: all(row[k] == v for k, v in selection.items())
    with metrics.open() as f:
        rows = [row for row in csv.DictReader(f) if selected(row)]
    with summary.open() as f:
        stats = {row["metric"]: row for row in csv.DictReader(f)
                 if selected(row) and row["metric"] in {"power", "fdp"}}
    assert len(rows) == len({r["repeat"] for r in rows}) == 50
    for metric in ["power", "fdp"]:
        assert abs(np.mean([float(r[metric]) for r in rows])-float(stats[metric]["mean"])) < 1e-12
    target = out / "source_data"
    target.mkdir(exist_ok=True)
    for name, data in [("complete_input_replicates.csv", rows), ("complete_input_summary.csv", list(stats.values()))]:
        with (target / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return stats


def main(out, metrics, summary):
    out.mkdir(parents=True, exist_ok=True)
    stats = validation(metrics, summary, out)
    W, H = 183, 144
    fig = plt.figure(figsize=(W/25.4, H/25.4), dpi=260, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
    ax.axis("off")
    labels = []

    def text(x, y, s, size=6, bold=False, ha="left", color=None, **kw):
        item = ax.text(x, y, s, size=size, weight="bold" if bold else "normal", ha=ha,
                       va="center", linespacing=1.15, color=color or C["ink"], zorder=9, **kw)
        labels.append(item)
        return item

    def line(x, y, col=None, lw=.6, **kw):
        ax.plot(x, y, c=col or C["edge"], lw=lw, solid_capstyle="round", **kw)

    def box(x, y, w, h, fill="white", edge=None, radius=1.5, lw=.6, **kw):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              fc=fill, ec=edge or C["edge"], lw=lw, **kw)
        ax.add_patch(patch)
        return patch

    def arrow(a, b, col=None, curve=0, lw=1.0, style="-|>", **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                    connectionstyle=f"arc3,rad={curve}",
                                    color=col or C["muted"], lw=lw, shrinkA=0, shrinkB=0, **kw))

    def panel(letter, title, x, y, w, h, accent, fill):
        box(x, y, w, h, fill, C["edge"], radius=2, lw=.65)
        text(x+2.8, y+h-4.3, letter, 8, True, color=accent)
        text(x+8, y+h-4.3, title, 7.2, True)
        line([x+3, x+w-3], [y+h-8, y+h-8], accent, .7)

    def matrix(x, y, w, h, a, palette, border=True):
        a = np.asarray(a)
        nr, nc = a.shape
        for i in range(nr):
            for j in range(nc):
                ax.add_patch(Rectangle((x+j*w/nc, y+(nr-i-1)*h/nr), w/nc, h/nr,
                                       fc=palette[int(a[i, j])], ec="white", lw=.2))
        if border:
            ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=C["edge"], lw=.35))

    def stack(x, y, w, h, a, palette):
        for dx, dy in [(1.2, 1.2), (.6, .6)]:
            ax.add_patch(Rectangle((x+dx, y+dy), w, h, fc=palette[0], ec=C["edge"], lw=.3))
        matrix(x, y, w, h, a, palette)

    def database(x, y, color):
        ax.add_patch(Rectangle((x, y), 4, 4.4, fc="white", ec=color, lw=.6))
        for cy in [y, y+2.2, y+4.4]:
            ax.add_patch(Ellipse((x+2, cy), 4, 1.4, fc="white", ec=color, lw=.6))

    def client(x, y, label, index):
        box(x, y, 34, 23, "white", C["blue_mid"], radius=1.5, lw=.75)
        ax.add_patch(Rectangle((x+.6, y+16.5), 32.8, 5.8, fc=C["blue_pale"], ec="none"))
        text(x+2, y+19.3, label, 6.5, True)
        database(x+27, y+17.6, C["blue"])
        a = (np.arange(30).reshape(6, 5)+index+np.arange(6)[:, None]) % 4
        stack(x+2.2, y+5.7, 8.1, 7.6, a, ["#D9E8F1", "#A0C4D9", "#6C9EBE", C["blue"]])
        b = (np.arange(18).reshape(6, 3)+index) % 3
        stack(x+13.2, y+5.7, 6.1, 7.6, b, ["#D4E7DB", "#92BEAA", C["green"]])
        for i in range(4):
            for j in range(4):
                ax.add_patch(Circle((x+24.2+1.7*i, y+6.5+1.7*j), .32, fc=C["blue"], ec="none"))
        line([x+23, x+23, x+31], [y+14, y+5.3, y+5.3], C["edge"], .4)
        text(x+6.4, y+2.5, "Counts", 5.8, ha="center")
        text(x+16.7, y+2.5, "Weights", 5.8, ha="center")
        text(x+27, y+2.5, "Space", 5.8, ha="center")

    panel("a", "Federated analysis across data holders", 2, 70, 105, 72, C["blue"], "#F4F8FB")
    client(7, 105, "Client 1", 0)
    client(7, 78, "Client 2", 1)
    client(67, 78, "Client M", 2)
    text(78.5, 130.6, "Coordinator", 6.7, True, ha="center")
    for dy in [0, 3.3, 6.6]:
        box(69, 116+dy, 19, 2.6, "#D8E7E9", C["green"], radius=.6, lw=.65)
        for j in range(3):
            ax.add_patch(Circle((71+j*1.5, 117.3+dy), .3, fc=C["green"], ec="none"))
        line([79, 85.7], [117.3+dy, 117.3+dy], C["green"], .5)
    text(78.5, 112.7, "Shared parameters", 6.1, ha="center")
    arrow((42, 118), (67.2, 120), C["green"], curve=-.10, lw=1.25)
    arrow((67.2, 124.5), (42, 122.5), C["blue"], curve=.12, lw=1.25)
    arrow((42, 88), (67.3, 115.7), C["green"], curve=.14, lw=1.25)
    arrow((66.3, 110.5), (42, 82.8), C["blue"], curve=-.14, lw=1.25)
    arrow((84.5, 102.5), (84.5, 109.3), C["green"], lw=1.25)
    arrow((77, 109.3), (77, 102.5), C["blue"], lw=1.25)
    text(49, 105.8, "Model\ninformation", 6, ha="center")
    text(54, 81.2, "...", 9, ha="center", color=C["muted"])
    text(54.5, 73.7, "Molecular matrices stay local during fitting", 6.1, True, ha="center")

    panel("b", "Shared statistical framework", 113, 70, 68, 72, C["green"], "#F5F9F6")
    text(147, 130, "Negative-binomial spatial model", 6.4, True, ha="center")
    text(147, 123.6, r"$\eta_{gk}(s)=\theta_{gk}+\beta_{gk}^{\mathsf{T}}h(s)$", 7, ha="center")
    for x, name in zip([119, 131, 143, 155, 167], ["Linear", "G1", "G2", "C1", "C2"]):
        u = np.linspace(-1, 1, 70)
        v = {"Linear": .5+.45*u, "G1": np.exp(-u*u/.5), "G2": np.exp(-u*u/.14),
             "C1": .5+.45*np.cos(np.pi*u), "C2": .5+.45*np.cos(2*np.pi*u)}[name]
        line([x, x, x+7.5], [119, 114, 114], C["edge"], .4)
        line(x+3.75*(u+1), 114+4.7*v, C["green"], .9)
        text(x+3.75, 111, name, 5.8, ha="center")
    text(147, 106.5, "Five families fitted separately", 5.8, ha="center")
    line([117, 177], [103, 103], C["edge"], .55)
    text(132, 99.8, "Sparse estimation", 6.1, True, ha="center")
    text(164, 99.8, "Independent refit", 6.1, True, ha="center")
    beta = [[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]
    stack(119, 86, 11, 8.4, beta, ["#E3EAED", C["blue"], C["coral"]])
    line([134, 145], [89, 89], C["edge"], .35)
    line([139.5, 139.5], [85, 93], C["edge"], .35)
    line([134, 138, 141, 145], [85.5, 89, 89, 93], C["blue"], .9)
    text(132, 82.3, "AP-FedAdam", 6.1, ha="center", color=C["blue"])
    cov = [[3, 1, 1, 0], [1, 3, 2, 1], [1, 2, 3, 1], [0, 1, 1, 3]]
    matrix(158.5, 86, 11, 8.4, cov, ["#ECF2EC", "#BFD9C9", "#84B6A1", C["green"]])
    text(164, 82.3, "Full covariance", 5.8, ha="center")
    line([149, 149], [81, 100.7], C["edge"], .5, linestyle=(0, (2, 2)))
    line([171.5, 175, 175, 127], [90, 90, 78.8, 78.8], C["green"], .7)
    arrow((127, 78.8), (127, 76), C["green"], lw=.7)
    text(127, 74.1, "Joint Wald", 5.8, ha="center")
    text(150, 74.1, "Cauchy", 5.8, ha="center")
    text(173, 74.1, "BH", 5.8, ha="center")
    arrow((137, 74.1), (141, 74.1), C["green"], lw=.7)
    arrow((160, 74.1), (166, 74.1), C["green"], lw=.7)
    arrow((107.8, 119), (112.2, 119), C["muted"], lw=.85)

    panel("c", "Integrated cell-type-specific SVG detection", 2, 4, 129, 60, C["gold"], "#FCFAF4")
    text(44, 52.2, r"Gene $g_1$: different cell-type patterns", 6.5, True, ha="center")
    text(29, 47.7, "Client 1", 6, ha="center")
    text(62, 47.7, "Client M", 6, ha="center")
    text(45.5, 47.7, "...", 6, ha="center")
    cmap = LinearSegmentedColormap.from_list("spatial_expression", ["#E4EDF2", "#669BB8", "#23577D"])

    def spatial_glyph(cx, cy, variable, variant):
        angle = np.linspace(0, 2*np.pi, 180)
        radius = 1+.08*np.cos(3*angle+variant)+.045*np.sin(5*angle)
        poly = np.column_stack((cx+10.3*radius*np.cos(angle), cy+6.7*radius*np.sin(angle)))
        ax.add_patch(Polygon(poly, fc="#EDF1EB", ec="#ABB8B7", lw=.5))
        inner = np.array([cx, cy]) + .87*(poly-np.array([cx, cy]))
        inside = MplPath(inner)
        for i in range(13):
            for j in range(9):
                px = -9.6+i*1.62+(j % 2)*.45
                py = -6+j*1.5
                if not inside.contains_point((cx+px, cy+py)):
                    continue
                value = .12+.81*(px+10)/20 if variable else .48
                ax.add_patch(Ellipse((cx+px, cy+py), 1.12, .93, angle=(i*19+j*13) % 90,
                                     fc=cmap(value), ec="white", lw=.14))

    for row, (cy, variable) in enumerate([(38, True), (21.5, False)]):
        text(9, cy, "Type A" if variable else "Type B", 6, ha="center", rotation=90)
        spatial_glyph(29, cy, variable, 0)
        spatial_glyph(62, cy, variable, 1)
    for j in range(40):
        ax.add_patch(Rectangle((31+j*.5, 9), .5, 1.8, fc=cmap(j/39), ec="none"))
    text(25.5, 9.8, "Low", 5.8, ha="center")
    text(57.5, 9.8, "High", 5.8, ha="center")
    text(77, 38, "Spatial", 5.8, color=C["blue"], rotation=90, ha="center")
    text(77, 21.5, "Constant", 5.8, color=C["muted"], rotation=90, ha="center")
    line([83, 83], [9, 53], C["edge"], .5)
    text(107, 51.8, "ctSVG candidates", 6.5, True, ha="center")
    text(110, 46.6, "Cell types", 5.8, ha="center")
    for x, s in zip([101, 111, 121], ["A", "B", "C"]):
        text(x, 42.6, s, 6, ha="center")
    calls = [[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]]
    matrix(96, 15, 30, 25, calls, ["#E6E9E4", C["green"]])
    for i in range(6):
        text(92.2, 40-(i+.5)*25/6, r"$g_"+str(i+1)+"$", 5.8, ha="center")
    text(107, 9.2, "Combined p and BH q", 5.8, ha="center")
    line([169, 169, 119], [69.5, 67, 67], C["green"], .8)
    arrow((119, 67), (119, 64.5), C["green"], lw=.8)

    panel("d", "Detection check", 135, 4, 46, 60, C["blue"], "#F7F9FA")
    text(158, 51.6, "Complete inputs", 6.1, True, ha="center")
    text(158, 47.5, "50 response replicates", 5.8, ha="center")
    for metric, y, xmax, title, color in [("power", 35.5, 1., "Power", C["blue"]),
                                         ("fdp", 18.5, .1, "Mean FDP", C["green"])]:
        mean, low, high = [float(stats[metric][k]) for k in ["mean", "lower", "upper"]]
        text(140, y+6.4, title, 6.1, True)
        text(176, y+6.4, f"{mean:.3f}", 6.1, True, ha="right", color=color)
        scale = lambda v: 142+33*v/xmax
        line([142, 175], [y-2.4, y-2.4], C["edge"], .45)
        for v in [0, xmax/2, xmax]:
            line([scale(v), scale(v)], [y-2.4, y-3.1], C["edge"], .45)
            text(scale(v), y-5.3, f"{v:g}", 5.8, ha="center")
        line([scale(low), scale(high)], [y, y], color, 1.2)
        for value in [low, high]:
            line([scale(value), scale(value)], [y-1.2, y+1.2], color, .7)
        ax.add_patch(Circle((scale(mean), y), .75, fc=color, ec="white", lw=.2))
    text(158, 7.3, "Means and 95% intervals", 5.8, ha="center", color=C["muted"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [t.get_window_extent(renderer) for t in labels]
    overlaps, outside = [], []
    for i, a in enumerate(bounds):
        if a.x0 < 0 or a.x1 > fig.bbox.x1 or a.y0 < 0 or a.y1 > fig.bbox.y1:
            outside.append(labels[i].get_text())
        for j in range(i+1, len(bounds)):
            b = bounds[j]
            if min(a.x1, b.x1)-max(a.x0, b.x0)>1 and min(a.y1, b.y1)-max(a.y0, b.y0)>1:
                overlaps.append([labels[i].get_text(), labels[j].get_text()])
    if overlaps or outside:
        raise RuntimeError(json.dumps({"text_overlaps": overlaps, "outside": outside}, indent=2))
    stem = out / "distrisvg_workflow_v5_candidate"
    for ext in ["pdf", "svg", "png", "tiff"]:
        fig.savefig(stem.with_suffix("."+ext), dpi=600 if ext=="tiff" else 280,
                    facecolor="white", metadata={"Title": "DistriSVG graphical candidate"} if ext=="pdf" else None)
    manifest = {"status": "Visual candidate, not integrated into manuscript", "size_mm": [W, H],
                "backend": "Python/Matplotlib", "minimum_font_pt": 5.8,
                "conceptual_panels": ["a", "b", "c"], "quantitative_panel": "d", "validation": stats,
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [metrics, summary]},
                "checks": {"text_overlap": "passed", "canvas_containment": "passed"},
                "script": Path(__file__).name, "online_sync": False}
    (out / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "calibration_metrics.csv")
    parser.add_argument("--summary", type=Path, default=RESULTS / "calibration_summary.csv")
    args = parser.parse_args()
    main(args.output_dir, args.metrics, args.summary)
