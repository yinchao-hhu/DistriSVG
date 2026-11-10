"""Generate manuscript-ready summaries only after complete stage outputs exist."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
OUT=ROOT/"04_数值实验/results/distrisvg_completion_2026-09-25"
TABLE=ROOT/"05_论文草稿/latex/tables/distrisvg_2026-09-25"
sys.path.insert(0,str(CODE.parent/"distrisvg_revision_2026_09_18"))
from metrics import pair_metrics, bh
from summarize_completion import latex_table


def real():
    metrics=pd.concat([pd.read_csv(OUT/"real_both_metrics.csv"),pd.read_csv(OUT/"real_weighted_metrics.csv")])
    assert len(metrics)==64
    metrics.to_csv(OUT/"real_all_metrics.csv",index=False)
    pairs=pd.concat([pd.read_csv(OUT/"real_both_pairs.csv"),pd.read_csv(OUT/"real_weighted_pairs.csv")])
    ranks=[]
    for (dataset,method,adj),part in pairs.groupby(["dataset","method","adjustment"]):
        none=part[part.offset=="none"].set_index(["gene_id","cell_type"])
        library=part[part.offset=="library"].set_index(["gene_id","cell_type"]).loc[none.index]
        assert none.index.is_unique and len(none)==(71100 if dataset=="PDAC" else 2100)
        top=lambda x:set(x.sort_values(["pair_p","gene_id","cell_type"],kind="stable").head(100).index)
        ranks.append(dict(dataset=dataset,method=method,adjustment=adj,
            spearman=float(spearmanr(none.pair_p,library.pair_p).statistic),
            top100_overlap=len(top(none)&top(library))))
    ranks=pd.DataFrame(ranks)
    ranks.to_csv(OUT/"real_full_offset_rank_sensitivity.csv",index=False)
    audit=pd.read_csv(OUT/"real_both_fit_audit.csv")
    assert len(audit)==50400
    audit.groupby(["dataset","offset","reason"],dropna=False).size().rename("fits").reset_index().to_csv(OUT/"real_fit_audit_summary.csv",index=False)
    names={"AP_diagonal":"AP diagonal", "refit_weighted_model":"Joint information", "refit_sandwich":"Joint HC1"}
    rows=[]
    for dataset in ("PDAC","Mouse"):
        for method,label in names.items():
            for offset in ("none","library"):
                r=metrics[(metrics.dataset==dataset)&(metrics.method==method)&(metrics.offset==offset)&(metrics.adjustment=="acat_bh")].iloc[0]
                rows.append([dataset,"None" if offset=="none" else "Library",label,int(r.selected_pairs),int(r.selected_genes)])
        choose=lambda method,offset:metrics[(metrics.dataset==dataset)&(metrics.method==method)&(metrics.offset==offset)&(metrics.adjustment=="acat_bh")].iloc[0]
        a,b=choose("refit_weighted_model","none"),choose("refit_weighted_model","library")
        c,d=choose("refit_sandwich","none"),choose("refit_sandwich","library")
        rank=ranks[(ranks.dataset==dataset)&(ranks.method=="refit_weighted_model")&(ranks.adjustment=="acat_bh")].iloc[0]
        text=(f"The full-panel canonical reanalysis evaluated {int(a.n_pairs):,} pairs with both offset choices. "
              f"The weight-consistent joint information refit selected {int(a.selected_pairs):,} pairs "
              f"({int(a.selected_genes):,} genes) without an offset and {int(b.selected_pairs):,} pairs "
              f"({int(b.selected_genes):,} genes) with a library offset under Cauchy--BH. "
              f"The corresponding HC1 counts were {int(c.selected_pairs):,} and {int(d.selected_pairs):,} pairs. "
              f"The information-refit rankings had offset-arm Spearman correlation {rank.spearman:.3f} "
              f"and shared {int(rank.top100_overlap)} of their top 100 pairs. "
              "These differences describe preprocessing and covariance sensitivity, not accuracy or validated "
              "false-discovery control; real-data truth and spatially calibrated uncertainty are unavailable "
              "(Supplementary Note~11).\n")
        if dataset=="Mouse":
            text+=("The mouse panel was selected for informative expression; its dense candidate calls do not "
                   "constitute genome-wide evidence of cell-type specificity.\n")
        (TABLE/f"real_{dataset.lower()}_summary.tex").write_text(text)
    latex_table(TABLE/"real_full.tex",["Dataset","Offset","Score / covariance","Pairs","Genes"],rows,"lllrr")
    bad=audit[~audit.valid]
    counts=bad.groupby("dataset").size().reindex(["PDAC","Mouse"],fill_value=0)
    weighted=pd.read_csv(OUT/"real_weighted_covariance_audit.csv")
    additional=weighted[~weighted.status.isin(["ok","invalid_original_refit"])].groupby("dataset").size().reindex(["PDAC","Mouse"],fill_value=0)
    weighted.groupby(["dataset","offset","status"]).size().rename("fits").reset_index().to_csv(OUT/"real_weighted_covariance_audit_summary.csv",index=False)
    section=(r"\begin{table}[htbp]\centering\small"+"\n"+
       r"\caption{\textbf{Full-panel sensitivity counts at the Cauchy--BH threshold.} Each PDAC arm tests 71,100 pairs; each mouse arm tests 2100. AP diagonal values are diagnostics. Joint information includes squared spot weights, and HC1 is an independent-spot sandwich. These are single real-data fits, without biological truth or replicate error bars.}"+"\n"+
       r"\label{tab:full-real}\input{tables/distrisvg_2026-09-25/real_full.tex}\end{table}"+"\n\n"+
       f"Across ten families/offset combinations per dataset, {int(counts.PDAC):,} of 47,400 PDAC "
       f"gene fits and {int(counts.Mouse):,} of 3000 mouse gene fits failed an interior, stationary, "
       "identifiable-refit check. They remain in the pair universe with conservative values. "
       f"The squared-weight information covariance check rejected an additional {int(additional.PDAC):,} PDAC "
       f"and {int(additional.Mouse):,} mouse gene fits; these receive flagged values of one only in that covariance arm. "
       "The source data list the reason for each fit and preserve all BH/BY and Bonferroni sensitivities.\n")
    (TABLE/"real_full_section.tex").write_text(section)


def competitors():
    base=OUT/"official_competitors"
    status=json.loads((base/"run_status.json").read_text())
    assert all(status[x]["exit_code"]==0 for x in ("CTSV","STANCE","CELINA","spVC"))
    seeded=json.loads((base/"spvc_seeded_audit/run_status.json").read_text())
    assert seeded["status"]=="identical_replay_verified"
    rows,frames=[],[]
    for method,key in (("CTSV","ctsv"),("STANCE","stance"),("Celina","celina"),("spVC","spvc")):
        for rep in range(3):
            conditions=["complete","missing"] if key=="ctsv" else ["complete"]
            for condition in conditions:
                directory=base/"spvc_seeded_audit/first" if key=="spvc" else base/"source_data"
                d=pd.read_csv(directory/f"repeat{rep}_{condition}_{key}_pairs.csv")
                assert len(d)==200 and not d.duplicated(["gene_id","cell_type"]).any()
                q=bh(d.pair_p)
                frames.append(d[["gene_id","cell_type","active","pair_p"]].assign(method=method,repeat=rep,condition=condition,pair_q=q))
                rows.append(dict(method=method,repeat=rep,condition=condition,source="fresh_official",**pair_metrics(d.active.to_numpy(),d.pair_p.to_numpy(),q)))
    old=pd.read_csv(ROOT/"04_数值实验/results/distrisvg_revision_2026-09-18/archive_competitor_pairs.csv")
    for (condition,rep),d in old[old.method=="ap_fedadam"].groupby(["condition","repeat"]):
        rows.append(dict(method="AP diagonal (archived)",repeat=rep,condition=condition,source="archived_AP",**pair_metrics(d.active.to_numpy(),d.pair_p.to_numpy())))
    refit=pd.read_csv(OUT/"archive_refit_common40_metrics.csv")
    for _,d in refit[(refit.adjustment=="acat_bh")&(refit.policy=="all")].iterrows():
        record=d.to_dict(); record["method"]="Joint refit / "+d.covariance
        record["source"]="new_joint_refit"; rows.append(record)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/"fresh_competitor_metrics.csv",index=False)
    pd.concat(frames).to_csv(OUT/"fresh_competitor_pairs.csv",index=False)
    mean=metrics.groupby(["condition","method"])[["power","precision","fdp","f1","auprc","auroc"]].agg(["mean","std"])
    mean.to_csv(OUT/"fresh_competitor_summary.csv")
    selected=[]
    for method in ("AP diagonal (archived)","Joint refit / model","Joint refit / sandwich","CTSV","STANCE","Celina","spVC"):
        r=mean.loc[("complete",method)]
        selected.append([method]+[f"${r[(x,'mean')]:.3f}\\pm{r[(x,'std')]:.3f}$" for x in ("power","fdp","auprc","auroc")])
    latex_table(TABLE/"fresh_competitors.tex",["Procedure","Power","FDP","AUPRC","AUROC"],selected,"lrrrr")
    text=(r"\begin{table}[htbp]\centering\scriptsize"+"\n"+
          r"\caption{\textbf{Fresh official executions and joint-refit checks on the common compact panel.} Mean $\pm$ s.d. over the same three complete-input datasets and 200 pairs per dataset. The AP diagonal row is the preserved archived diagnostic; four comparator rows are fresh official-source executions. Joint-refit rows fit all 160 genes and then re-adjust on the fixed 40-gene evaluation panel. No parameters are selected from these outcomes.}"+"\n"+
          r"\label{tab:fresh-competitors}\input{tables/distrisvg_2026-09-25/fresh_competitors.tex}\end{table}"+"\n\n"+
          "The two refit covariance forms use the same unpenalised estimates; they are not additional independently optimised sparse methods. "
          "Each gene is refitted with its full baseline and dispersion nuisance parameters before evaluation-panel filtering. The common-panel BH adjustment is recalculated on 200 pairs "
          "and must not be compared with the 800-pair adjustment without stating the universe. The spVC row uses a fixed-seed official run, "
          "verified by an identical replay; unseeded attempts remain in the audit. Fresh missing-input CTSV outputs, axis-level raw values "
          "and every external exit code are included in the source data.\n")
    (TABLE/"competitor_section.tex").write_text(text)


def official():
    base=OUT/"official_mouse"
    extra=base/"mmm/run_audit_posterior.csv"
    if extra.exists():
        audit_path=base/"mmm/run_audit.csv"
        attempts=pd.concat([pd.read_csv(audit_path),pd.read_csv(extra)],ignore_index=True)
        completed=attempts[attempts.status=="completed"].drop_duplicates("section",keep="last")
        assert set(completed.section)=={"Anterior","Posterior","Anterior2"}
        assert all((base/"mmm"/f"{section}_done.rds").exists() for section in completed.section)
        history=base/"mmm/run_attempt_audit.csv"
        if history.exists(): attempts=pd.concat([pd.read_csv(history),attempts]).drop_duplicates()
        attempts.to_csv(history,index=False)
        completed.to_csv(audit_path,index=False)
    rctd=[]; calls=[]
    for stage in ("rctd","cside","mmm"):
        audit=pd.read_csv(base/stage/"run_audit.csv")
        assert set(audit.section)=={"Anterior","Posterior","Anterior2"} and (audit.status=="completed").all(),stage
        for section in ("Anterior","Posterior","Anterior2"):
            if stage=="rctd":
                weights=pd.read_csv(base/stage/f"{section}_weights.csv",index_col=0)
                before=pd.read_csv(base/"inputs"/section/"spots.csv")
                agreement=pd.read_csv(base/stage/f"{section}_nnls_agreement.csv")
                assert np.isfinite(weights).all().all() and np.max(np.abs(weights.sum(1)-1))<1e-10
                rctd.append(dict(section=section,input_spots=len(before),retained_spots=len(weights),omitted_spots=len(before)-len(weights),
                                 min_spearman=agreement.spearman.min(),max_spearman=agreement.spearman.max()))
            else:
                d=pd.read_csv(base/stage/f"{section}_pairs.csv")
                assert len(d)==2100 and not d.duplicated(["gene_id","cell_type"]).any()
                assert np.allclose(d.pair_bh_q,bh(d.pair_p),atol=1e-12)
                selected=d.pair_bh_q<.05
                calls.append(dict(section=section,method="C-SIDE" if stage=="cside" else "MMM",pairs_returned=int((d.status=="returned").sum()),
                    pairs_not_returned=int((d.status!="returned").sum()),selected_pairs=int(selected.sum()),selected_genes=d.loc[selected,"gene_id"].nunique()))
    rctd=pd.DataFrame(rctd); calls=pd.DataFrame(calls)
    rctd.to_csv(OUT/"official_rctd_summary.csv",index=False); calls.to_csv(OUT/"official_mouse_detection_summary.csv",index=False)
    latex_table(TABLE/"official_mouse.tex",["Section","Method","Returned pairs","Selected pairs","Genes"],
                calls[["section","method","pairs_returned","selected_pairs","selected_genes"]].values.tolist(),"llrrr")
    latex_table(TABLE/"rctd.tex",["Section","Input spots","Retained spots","NNLS Spearman range"],
        [[r.section,int(r.input_spots),int(r.retained_spots),f"{r.min_spearman:.3f}--{r.max_spearman:.3f}"] for r in rctd.itertuples()],"lrrr")
    text=(f"Formal RCTD deconvolution retained {int(rctd.retained_spots.sum()):,} of 8875 input spots after its default quality filters. "
          f"Across section--cell-type comparisons, Spearman agreement with the original NNLS proportions ranged from "
          f"{rctd.min_spearman.min():.3f} to {rctd.max_spearman.max():.3f}. "
          "Separate C-SIDE and MMM screens used the original NNLS proportions and the fixed target panel, retaining each method's eligibility filters. "
          "Their per-section candidate counts and excluded pairs are reported in Supplementary Note~12. "
          "These comparisons neither identify a true cell composition nor register corresponding anatomical regions.\n")
    (TABLE/"official_mouse_summary.tex").write_text(text)
    section=(r"\begin{table}[htbp]\centering\small"+"\n"+
        r"\caption{\textbf{Official RCTD full-mode output coverage and NNLS agreement.} Spearman ranges are across seven broad classes within a section and do not measure accuracy.}"+"\n"+
        r"\label{tab:rctd-official}\input{tables/distrisvg_2026-09-25/rctd.tex}\end{table}"+"\n\n"+
        r"\begin{table}[htbp]\centering\small"+"\n"+
        r"\caption{\textbf{Official reference-informed screens on the mouse target panel.} Each section retains a 2100-pair reporting universe. Returned pairs passed the method's filters and produced valid values; other pairs remain explicitly flagged with reporting value one. One BH correction is applied over the full reporting universe for each method and section. Gene counts are unique genes among selected pairs, not tested-pair counts.}"+"\n"+
        r"\label{tab:official-mouse}\input{tables/distrisvg_2026-09-25/official_mouse.tex}\end{table}"+"\n\n"+
        "The official gene-symbol aggregation affects Ccl27a within the 300-gene target panel; the other 299 target genes have unique symbols. "
        "Thus the comparison also retains a documented one-gene extraction difference, in addition to method-specific filtering and model differences.\n")
    (TABLE/"official_mouse_section.tex").write_text(section)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["real","competitors","official"])
    args=parser.parse_args(); TABLE.mkdir(exist_ok=True,parents=True)
    {"real":real,"competitors":competitors,"official":official}[args.stage]()
