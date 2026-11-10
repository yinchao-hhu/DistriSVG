"""Matched pooled replay of the frozen complete-input inference calibration."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
CORE = CODE.parent / 'distrisvg_completion_2026_09_25'
sys.path.insert(0, str(CORE))
import run_completion as original
from joint_refit import LocalGene, aggregate, gene_sites, local_summary, refit_gene
from metrics import pair_metrics

OLD = original.OUT
OUT = ROOT / '04_数值实验/results/workflow_pooled_2026-10-26'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool(sites):
    if not all(site.weight == 1 for site in sites):
        raise ValueError('This paired calibration requires equal site sizes and unit weights')
    return LocalGene(np.concatenate([s.x for s in sites]),
                     np.concatenate([s.y for s in sites]),
                     np.concatenate([s.offset for s in sites]))


def fit_pooled(sites, k, gene):
    return refit_gene([pool(gene_sites(sites, gene))], k)


def load_inputs(rep):
    base, _, types, _ = original.old.load_sim(0, 'complete')
    path = OLD / 'calibration' / f'sparse_{rep:03d}_input.npz'
    with np.load(path) as data:
        y, beta = data['y'], data['beta']
    assert y.shape == (300, 40) and beta.shape == (2, 5, 40)
    clients, start = [], 0
    for c in base:
        n = len(c['y'])
        count = y[start:start+n].copy()
        clients.append(dict(c, y=count, library=count.sum(1)))
        start += n
    assert start == len(y)
    active = np.any(beta != 0, axis=0).T.ravel().astype(int)
    return clients, types, active, path


def run_repeat(rep, workers):
    clients, types, active, path = load_inputs(rep)
    k, g = len(types), 40
    p_local, p_pooled, fits, comparisons = [], [], [], []
    hashes = {str(path.relative_to(ROOT)): sha(path)}
    checkpoint = OUT / 'refits'
    checkpoint.mkdir(parents=True, exist_ok=True)
    core_hash = sha(CORE / 'joint_refit.py')
    for family in original.old.FAMILIES:
        sites, _ = original.old.make_sites(clients, family, 'none')
        prefix = OLD / 'refits' / f'calib_sparse_r{rep}_complete_{family}_g0000-0040'
        metadata = json.loads(prefix.with_suffix('.json').read_text())['identity']
        assert metadata['core_sha256'] == core_hash
        assert metadata['input_sha256'] == original.array_hash(sites)
        for ext in ['.json','.npz','.csv']:
            f = prefix.with_suffix(ext)
            hashes[str(f.relative_to(ROOT))] = sha(f)
        old_fit = np.load(prefix.with_suffix('.npz'))
        p_local.append(old_fit['p_model'])
        local_gene = gene_sites(sites, 0)
        summed = aggregate(old_fit['parameter'][0], local_gene, second=True)
        joined = local_summary(old_fit['parameter'][0], pool(local_gene), second=True)
        identity_error = []
        for a, b in zip(summed, joined):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9)
            identity_error.append(float(np.max(np.abs(a-b))))
        new_path = checkpoint / f'pooled_sparse_{rep:03d}_{family}.npz'
        new_meta = new_path.with_suffix('.json')
        new_audit = new_path.with_suffix('.csv')
        identity = dict(input_sha256=original.array_hash(sites), core_sha256=core_hash,
                        initialization='original default, no warm start', pooled=True)
        if new_path.exists():
            assert json.loads(new_meta.read_text()) == identity
            pooled_fit = np.load(new_path)
            audit = pd.read_csv(new_audit)
        else:
            results = Parallel(n_jobs=workers)(delayed(fit_pooled)(sites,k,gene) for gene in range(g))
            values = {key:np.stack([item[key] for item in results])
                      for key in ['parameter','p_model','p_sandwich']}
            np.savez_compressed(new_path, **values)
            audit = pd.DataFrame([{name:value for name,value in item.items()
                                   if name not in values} for item in results])
            audit.insert(0,'gene_id',[f'calib_{i:03d}' for i in range(g)])
            audit.to_csv(new_audit,index=False)
            new_meta.write_text(json.dumps(identity,indent=2))
            pooled_fit = values
        if 'gene_id' not in audit:
            audit.insert(0,'gene_id',[f'calib_{i:03d}' for i in range(g)])
            audit.to_csv(new_audit,index=False)
        archived_audit = pd.read_csv(prefix.with_suffix('.csv'))
        fits.append(audit.assign(repeat=rep,family=family))
        p_pooled.append(pooled_fit['p_model'])
        comparisons.append(dict(repeat=rep,family=family,
            max_parameter_difference=float(np.max(np.abs(pooled_fit['parameter']-old_fit['parameter']))),
            max_family_p_difference=float(np.max(np.abs(pooled_fit['p_model']-old_fit['p_model']))),
            status_disagreements=int((audit.reason.to_numpy()!=archived_audit.reason.to_numpy()).sum()),
            loss_identity_error=identity_error[0], gradient_identity_error=identity_error[1],
            hessian_identity_error=identity_error[2], meat_identity_error=identity_error[3]))
    rows, pairs = [], []
    old_metrics = pd.read_csv(OLD/'calibration'/f'sparse_{rep:03d}_metrics.csv')
    old_row = old_metrics[(old_metrics.policy=='complete') &
                         (old_metrics.covariance=='model') & (old_metrics.adjustment=='acat_bh')].iloc[0]
    for method, values in [('DistriSVG',p_local),('Pooled',p_pooled)]:
        p,q = original.combine(values)['acat_bh']
        result = pair_metrics(active,p,q)
        if method == 'DistriSVG':
            for name in ['tp','fp','fn','tn','power','fdp']:
                np.testing.assert_allclose(result[name],old_row[name],atol=1e-14)
        rows.append(dict(method=method,repeat=rep,**result))
        pairs.append(pd.DataFrame(dict(method=method,repeat=rep,
            gene_id=np.repeat([f'calib_{i:03d}' for i in range(g)],k),
            cell_type=np.tile(types,g),active=active,pair_p=p,pair_q=q)))
    return pd.DataFrame(rows),pd.concat(pairs),pd.concat(fits),pd.DataFrame(comparisons),hashes


def summarize(frame, pairs):
    rng = np.random.default_rng(20261026)
    n = frame['repeat'].nunique()
    bootstrap = rng.integers(0,n,(20000,n))
    rows, differences = [], []
    for metric in ['power','fdp']:
        wide = frame.pivot(index='repeat',columns='method',values=metric).sort_index()
        for method in ['DistriSVG','Pooled']:
            values = wide[method].to_numpy()
            lo,hi = np.quantile(values[bootstrap].mean(1),[.025,.975])
            rows.append(dict(metric=metric,method=method,n_repeats=n,mean=float(values.mean()),
                sd=float(values.std(ddof=1)),lower=float(lo),upper=float(hi),
                interval='95% paired response-replicate percentile bootstrap, 20000 resamples'))
        difference = (wide.DistriSVG-wide.Pooled).to_numpy()
        lo,hi = np.quantile(difference[bootstrap].mean(1),[.025,.975])
        differences.append(dict(metric=metric,mean_difference=float(difference.mean()),
                                lower=float(lo),upper=float(hi)))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False)
    q = pairs.pivot(index=['repeat','gene_id','cell_type'],columns='method',values='pair_q')
    verification = dict(n_repeats=n,pooled_gene_family_fits=n*40*5,
        compared_pairs=len(q),discordant_discoveries=int(((q.DistriSVG<.05)!=(q.Pooled<.05)).sum()),
        max_combined_q_difference=float(np.max(np.abs(q.DistriSVG-q.Pooled))),
        complete=n==50,not_an_optimizer_comparison=True,
        scope='Complete-input fixed-design inference replay, not new null calibration or network deployment')
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    (OUT/'figure_values.json').write_text(json.dumps(rows,indent=2))
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=50)
    args = parser.parse_args()
    assert 1 <= args.repeats <= 50 and 1 <= args.workers <= 6
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    metrics,pairs,audits,comparisons,inputs=[],[],[],[],{}
    for rep in range(args.repeats):
        m,p,a,c,h = run_repeat(rep,args.workers)
        metrics.append(m);pairs.append(p);audits.append(a);comparisons.append(c);inputs.update(h)
        pd.concat(metrics).to_csv(OUT/'metrics.csv',index=False)
        print(f'Replicate {rep+1}/{args.repeats}; pooled fits {(rep+1)*200}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    metrics,pairs = pd.concat(metrics),pd.concat(pairs)
    pairs.to_csv(OUT/'pairs.csv',index=False)
    pd.concat(audits).to_csv(OUT/'fit_audit.csv',index=False)
    pd.concat(comparisons).to_csv(OUT/'numerical_equivalence.csv',index=False)
    inputs.update(original.old.INPUTS)
    assert all(sha(ROOT/name)==digest for name,digest in inputs.items())
    manifest=dict(date='2026-10-26',source_inputs=inputs,
        core_sha256=sha(CORE/'joint_refit.py'),runner_sha256=sha(__file__),
        config=dict(repeats=args.repeats,sites=3,spots_per_site=100,genes=40,cell_types=5,
            families=list(original.old.FAMILIES),q=.05,policy='complete',offset='none',
            covariance='observed full nuisance-adjusted information',combination='ACAT then BH',
            failures='p=1 retained',same_initialization=True),
        distributed_arm='Frozen 2026-09-25 local-summary fits, hashes and metrics checked',
        pooled_arm='New optimization on vertically concatenated data from the same inputs',
        packages=dict(numpy=np.__version__,pandas=pd.__version__),seconds=time.perf_counter()-started)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summarize(metrics,pairs),indent=2),flush=True)


if __name__=='__main__':
    main()
