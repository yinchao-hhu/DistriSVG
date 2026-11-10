"""Traceable tables and calibration figure, without selecting favorable arms."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25"
FIG = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-09-25"
TABLE = ROOT / "05_论文草稿/latex/tables/distrisvg_2026-09-25"
sys.path.insert(0, str(CODE.parent / "estimation"))
from metrics import pair_metrics, bh

POLICIES = ["complete", "missing_all", "missing_complete_sites"]
LABELS = {"complete": "Complete", "missing_all": "Omitted / all sites",
          "missing_complete_sites": "Omitted / complete sites"}


def interval(values, rng):
    values = np.asarray(values, float)
    sample = rng.choice(values, size=(20000, len(values)), replace=True).mean(1)
    return np.quantile(sample, [.025, .975])


def latex_table(path, columns, rows, formats):
    text = [r"\begin{tabular}{" + formats + "}", r"\toprule",
            " & ".join(columns) + r" \\", r"\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(text) + "\n")


def calibration():
    data = pd.read_csv(OUT / "calibration_metrics.csv")
    data["signal"] = data.signal.fillna("null")
    assert len(data) == 4500
    assert data.groupby("signal")["repeat"].nunique().to_dict() == {"null": 100, "sparse": 50}
    rows = []
    rng = np.random.default_rng(20260925)
    for keys, group in data.groupby(["signal", "policy", "covariance", "adjustment"], sort=True):
        signal, policy, cov, adj = keys
        for metric in (["any_discovery", "null_p_below05"] if signal == "null" else ["fdp", "power", "auprc", "auroc"]):
            values = group[metric].to_numpy(float)
            if metric == "any_discovery":
                result = binomtest(int(values.sum()), len(values)).proportion_ci(method="exact")
                lo, hi = result.low, result.high
                definition = "Clopper-Pearson 95%"
            else:
                lo, hi = interval(values, rng)
                definition = "percentile 95%, 20000 response-replicate resamples"
            rows.append(dict(signal=signal, policy=policy, covariance=cov, adjustment=adj,
                             metric=metric, n_repeats=len(values), mean=values.mean(),
                             sd=values.std(ddof=1), lower=lo, upper=hi, interval=definition))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    audits = []
    for path in sorted((OUT / "calibration").glob("*_audit.csv")):
        item = pd.read_csv(path)
        item["signal"] = item.signal.fillna("null")
        audits.append(item)
    audit = pd.concat(audits, ignore_index=True)
    assert len(audit) == 90000
    counts = audit.groupby(["signal", "policy", "family", "reason"], dropna=False).size().rename("fits").reset_index()
    counts.to_csv(OUT / "calibration_fit_audit_summary.csv", index=False)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.right": False,
                        "axes.spines.top": False, "axes.linewidth": .6, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 81/25.4))
    specs = [("model", "acat_bh", "Joint model + BH", "#366A8A", "o"),
             ("model", "acat_by", "Joint model + BY", "#83ACBF", "s"),
             ("sandwich", "acat_bh", "HC1 + BH", "#9C5368", "o"),
             ("sandwich", "acat_by", "HC1 + BY", "#CCA0A9", "s")]
    source = []
    for index, (ax, signal, metric, title) in enumerate(zip(axes, ["null", "sparse", "sparse"],
            ["any_discovery", "fdp", "power"], ["Global-null error", "Sparse-signal FDP", "Sparse-signal power"])):
        for j, (cov, adj, label, color, marker) in enumerate(specs):
            part = summary[(summary.signal==signal) & (summary.metric==metric) &
                           (summary.covariance==cov) & (summary.adjustment==adj)].set_index("policy").loc[POLICIES]
            x = np.arange(3)+(j-1.5)*.13
            y = part["mean"].to_numpy()
            ax.errorbar(x, y, yerr=np.vstack([y-part.lower, part.upper-y]), color=color,
                        marker=marker, markersize=3.5, linewidth=.8, capsize=2, label=label, linestyle="none")
            source.append(part.reset_index().assign(panel="abc"[index]))
        ax.set_xticks(range(3), ["Complete", "Omitted\nretained", "Omitted\nexcluded"])
        ax.tick_params(axis="both", length=3, width=.6)
        ax.set_title(title, pad=10)
        ax.text(-.16, 1.07, "abc"[index], transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_xlim(-.5, 2.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=.4)
        if metric != "power":
            ax.axhline(.05,color="#777777",ls="--",lw=.7)
        ax.set_ylim((0, .26) if signal == "null" else ((0,.43) if metric=="fdp" else (.75,1.015)))
    axes[0].set_ylabel("Probability of any discovery")
    axes[1].set_ylabel("Mean false discovery proportion")
    axes[2].set_ylabel("Mean true positive rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",ncol=4,bbox_to_anchor=(.5,.005),columnspacing=1.6,handletextpad=.35)
    fig.subplots_adjust(left=.075,right=.99,top=.86,bottom=.28,wspace=.54)
    for ext in ("pdf", "svg", "png", "tiff"):
        fig.savefig(FIG / ("refit_calibration."+ext),dpi=600 if ext=="tiff" else 200,facecolor="white")
    plt.close(fig)
    pd.concat(source).to_csv(FIG / "refit_calibration_source.csv",index=False)
    selected = []
    for policy in POLICIES:
        for cov, adj, label, _, _ in specs:
            def row(signal, metric):
                return summary[(summary.signal==signal)&(summary.policy==policy)&(summary.covariance==cov)&
                               (summary.adjustment==adj)&(summary.metric==metric)].iloc[0]
            null, fdp, power = row("null","any_discovery"),row("sparse","fdp"),row("sparse","power")
            selected.append([LABELS[policy],label,f"{null['mean']:.3f} [{null.lower:.3f}, {null.upper:.3f}]",
                             f"{fdp['mean']:.3f}",f"{power['mean']:.3f}"])
    latex_table(TABLE / "calibration.tex", ["Input policy","Covariance / rule","Null error [95\\% CI]","FDP","Power"],selected,"llrrr")
    print(summary[(summary.covariance.isin(["model","sandwich"]))&(summary.adjustment=="acat_bh")&
                  (summary.metric.isin(["any_discovery","fdp","power"]))].to_string(index=False))


def archived_common():
    panel = pd.read_csv(ROOT / "05_论文草稿/latex/figs/simulation_method_comparison/source_data/official_method_gene_panel.csv").gene_id
    data = pd.read_csv(OUT / "archive_refit_pairs.csv")
    data = data[data.gene_id.isin(panel) & data.condition.isin(["complete","missing"])]
    rows, frames = [], []
    for keys, group in data.groupby(["condition","policy","covariance","adjustment","repeat"]):
        condition, policy, cov, adj, repeat = keys
        assert len(group)==200
        q = bh(group.pair_p)
        if adj.endswith("_by"):
            q = np.minimum(1,q*np.sum(1/np.arange(1,201)))
        rows.append(dict(condition=condition,policy=policy,covariance=cov,adjustment=adj,repeat=repeat,
                         **pair_metrics(group.active.to_numpy(),group.pair_p.to_numpy(),q)))
        frames.append(group.assign(pair_q=q))
    pd.DataFrame(rows).to_csv(OUT / "archive_refit_common40_metrics.csv",index=False)
    pd.concat(frames).to_csv(OUT / "archive_refit_common40_pairs.csv",index=False)


def main():
    FIG.mkdir(parents=True,exist_ok=True); TABLE.mkdir(parents=True,exist_ok=True)
    calibration()
    archived_common()


if __name__ == "__main__":
    main()
