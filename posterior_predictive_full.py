"""
Per-condition extension of posterior_predictive.py.

posterior_predictive.py reports only the aggregate super-additivity index
distribution (R_combo - R_indep). Paper 2's Table 2 and Figures 1-2 need the
full per-condition breakdown -- E_final for all 8 factorial conditions,
fractional reduction R for each single-agent and pairwise condition, and the
Bliss-independence prediction and super-additivity gap for each of the three
pairwise combinations (ARI+MEM, ARI+BUS, MEM+BUS), not just the triple
combination.

This does NOT re-run MCMC -- it resamples from the existing saved traces
(cct_prior.nc/cct_posterior.nc, cct_prior_v2.nc/cct_posterior_v2.nc) and
re-runs only the forward ODE factorial simulation per draw, exactly like
posterior_predictive.py does for the aggregate index.
"""

import csv
import numpy as np
import arviz as az

from priors import STRUCTURAL, IDENTIFIED_PARAM_NAMES, scipy_frozen_dists
import cct_model as cm

SEED = 20260712
N_DRAWS = 800

# v1 (MEM-only likelihood) traces only sampled the params identified by that
# likelihood -- phi1/phi3/delta_BUS were provably unidentified and excluded
# from the MCMC (see priors.py history / REPORT.md). When the factorial
# needs an ARI or BUS dose (v1 traces lack those columns), backfill from
# their own literature-range priors, exactly as the original v1
# posterior_predictive.py did.
_PRIOR_DISTS = scipy_frozen_dists()

CONDITIONS = [
    ('vehicle',      (False, False, False)),
    ('ARI_only',     (True,  False, False)),
    ('MEM_only',     (False, True,  False)),
    ('BUS_only',     (False, False, True)),
    ('ARI_MEM',      (True,  True,  False)),
    ('ARI_BUS',      (True,  False, True)),
    ('MEM_BUS',      (False, True,  True)),
    ('triple_combo', (True,  True,  True)),
]


def param_dict_from_row(row, rng, missing_names):
    p = dict(STRUCTURAL)
    p.update(row)
    for name in missing_names:
        p[name] = float(_PRIOR_DISTS[name].rvs(random_state=rng))
    return p


def run_full_distribution(idata_path, group, n_draws, rng):
    idata = az.from_netcdf(idata_path)
    ds = idata[group].to_dataset()
    available = set(ds.data_vars)
    present_names = [n for n in IDENTIFIED_PARAM_NAMES if n in available]
    missing_names = [n for n in IDENTIFIED_PARAM_NAMES if n not in available]
    if missing_names:
        print(f"  backfilling from literature-range priors (not in this trace): {missing_names}")
    flat = ds.stack(sample=("chain", "draw"))
    n_total = flat.sizes["sample"]
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

    # per-condition E_final draws
    E = {name: [] for name, _ in CONDITIONS}
    failures = 0
    for i in idx:
        row = {name: float(flat[name].values[i]) for name in present_names}
        p = param_dict_from_row(row, rng, missing_names)
        try:
            results = cm.run_factorial(p)
        except RuntimeError:
            failures += 1
            continue
        for name, key in CONDITIONS:
            E[name].append(results[key])

    E = {name: np.array(vals) for name, vals in E.items()}
    n_used = len(E['vehicle'])

    # derived fractional reductions R_X = (E_ctrl - E_X) / E_ctrl, per-draw
    E_ctrl = E['vehicle']
    R = {}
    for name, _ in CONDITIONS:
        if name == 'vehicle':
            continue
        R[name] = (E_ctrl - E[name]) / E_ctrl

    # Bliss-independence predictions and super-additivity gaps, per-draw
    R_indep_AM = 1 - (1 - R['ARI_only']) * (1 - R['MEM_only'])
    R_indep_AB = 1 - (1 - R['ARI_only']) * (1 - R['BUS_only'])
    R_indep_MB = 1 - (1 - R['MEM_only']) * (1 - R['BUS_only'])
    R_indep_combo = 1 - (1 - R['ARI_only']) * (1 - R['MEM_only']) * (1 - R['BUS_only'])

    sa_AM = R['ARI_MEM'] - R_indep_AM
    sa_AB = R['ARI_BUS'] - R_indep_AB
    sa_MB = R['MEM_BUS'] - R_indep_MB
    sa_combo = R['triple_combo'] - R_indep_combo

    return {
        'E': E, 'R': R, 'n_used': n_used, 'n_failures': failures,
        'R_indep_AM': R_indep_AM, 'R_indep_AB': R_indep_AB,
        'R_indep_MB': R_indep_MB, 'R_indep_combo': R_indep_combo,
        'sa_AM': sa_AM, 'sa_AB': sa_AB, 'sa_MB': sa_MB, 'sa_combo': sa_combo,
    }


def summarize_arr(arr):
    hdi = az.hdi(arr, prob=0.94)
    return {
        'mean': float(np.mean(arr)), 'sd': float(np.std(arr)),
        'hdi_low': float(hdi[0]), 'hdi_high': float(hdi[1]),
    }


def main(posterior_path, out_suffix):
    rng = np.random.default_rng(SEED)
    print(f"Running full per-condition posterior-predictive simulation "
          f"({N_DRAWS} draws) for {posterior_path} ...")
    res = run_full_distribution(posterior_path, 'posterior', N_DRAWS, rng)
    print(f"  n_used={res['n_used']}  n_failures={res['n_failures']}")

    rows = []
    for name, _ in CONDITIONS:
        s = summarize_arr(res['E'][name])
        rows.append(['E_final', name, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])
    for name in ['ARI_only', 'MEM_only', 'BUS_only', 'ARI_MEM', 'ARI_BUS', 'MEM_BUS', 'triple_combo']:
        s = summarize_arr(res['R'][name])
        rows.append(['R_reduction', name, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])
    for label, arr in [('R_indep_AM', res['R_indep_AM']), ('R_indep_AB', res['R_indep_AB']),
                        ('R_indep_MB', res['R_indep_MB']), ('R_indep_combo', res['R_indep_combo']),
                        ('super_additivity_AM', res['sa_AM']), ('super_additivity_AB', res['sa_AB']),
                        ('super_additivity_MB', res['sa_MB']), ('super_additivity_combo', res['sa_combo'])]:
        s = summarize_arr(arr)
        rows.append(['derived', label, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])

    out_path = f'results/full_condition_summary{out_suffix}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'name', 'mean', 'sd', 'hdi94_low', 'hdi94_high'])
        w.writerows(rows)
    print(f"Saved: {out_path}")
    return rows


if __name__ == '__main__':
    import sys
    if '--v2' in sys.argv:
        main('results/cct_posterior_v2.nc', '_v2')
    else:
        main('results/cct_posterior.nc', '')
