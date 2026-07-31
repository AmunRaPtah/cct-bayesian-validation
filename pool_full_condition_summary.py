"""
Pool the two independent-seed v1 long-chain posterior-predictive runs
(cct_posterior_v1_longrun.nc, seed 20260712; cct_posterior_v1_confirm.nc,
seed 20260713) into a single n=1600 per-condition summary, replacing
Table 2's original short-run numbers. Reuses posterior_predictive_full.py's
per-draw simulation logic but pools raw draws across both traces before
summarizing, rather than summarizing each trace separately.
"""
import numpy as np
import arviz as az

from priors import STRUCTURAL, IDENTIFIED_PARAM_NAMES, scipy_frozen_dists
import cct_model as cm
from posterior_predictive_full import CONDITIONS, param_dict_from_row

SEED = 20260715
N_DRAWS_PER_RUN = 800


def collect_raw(idata_path, n_draws, rng):
    idata = az.from_netcdf(idata_path)
    ds = idata['posterior'].to_dataset()
    available = set(ds.data_vars)
    present_names = [n for n in IDENTIFIED_PARAM_NAMES if n in available]
    missing_names = [n for n in IDENTIFIED_PARAM_NAMES if n not in available]
    flat = ds.stack(sample=("chain", "draw"))
    n_total = flat.sizes["sample"]
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

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
    return E, failures


def summarize_arr(arr):
    arr = np.array(arr)
    hdi = az.hdi(arr, prob=0.94)
    return {
        'mean': float(np.mean(arr)), 'sd': float(np.std(arr)),
        'hdi_low': float(hdi[0]), 'hdi_high': float(hdi[1]),
    }


def main():
    rng = np.random.default_rng(SEED)
    E1, fail1 = collect_raw('results/cct_posterior_v1_longrun.nc', N_DRAWS_PER_RUN, rng)
    E2, fail2 = collect_raw('results/cct_posterior_v1_confirm.nc', N_DRAWS_PER_RUN, rng)
    print(f"Run1 failures={fail1}  Run2 failures={fail2}")

    E_pooled = {name: np.array(E1[name] + E2[name]) for name, _ in CONDITIONS}
    n_used = len(E_pooled['vehicle'])
    print(f"Pooled n_used={n_used}")

    E_ctrl = E_pooled['vehicle']
    R = {}
    for name, _ in CONDITIONS:
        if name == 'vehicle':
            continue
        R[name] = (E_ctrl - E_pooled[name]) / E_ctrl

    R_indep_AM = 1 - (1 - R['ARI_only']) * (1 - R['MEM_only'])
    R_indep_AB = 1 - (1 - R['ARI_only']) * (1 - R['BUS_only'])
    R_indep_MB = 1 - (1 - R['MEM_only']) * (1 - R['BUS_only'])
    R_indep_combo = 1 - (1 - R['ARI_only']) * (1 - R['MEM_only']) * (1 - R['BUS_only'])

    sa_AM = R['ARI_MEM'] - R_indep_AM
    sa_AB = R['ARI_BUS'] - R_indep_AB
    sa_MB = R['MEM_BUS'] - R_indep_MB
    sa_combo = R['triple_combo'] - R_indep_combo

    import csv
    rows = []
    for name, _ in CONDITIONS:
        s = summarize_arr(E_pooled[name])
        rows.append(['E_final', name, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])
    for name in ['ARI_only', 'MEM_only', 'BUS_only', 'ARI_MEM', 'ARI_BUS', 'MEM_BUS', 'triple_combo']:
        s = summarize_arr(R[name])
        rows.append(['R_reduction', name, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])
    for label, arr in [('R_indep_AM', R_indep_AM), ('R_indep_AB', R_indep_AB),
                        ('R_indep_MB', R_indep_MB), ('R_indep_combo', R_indep_combo),
                        ('super_additivity_AM', sa_AM), ('super_additivity_AB', sa_AB),
                        ('super_additivity_MB', sa_MB), ('super_additivity_combo', sa_combo)]:
        s = summarize_arr(arr)
        rows.append(['derived', label, s['mean'], s['sd'], s['hdi_low'], s['hdi_high']])
        print(f"{label}: mean={s['mean']*100:.2f}pp  hdi=[{s['hdi_low']*100:.2f},{s['hdi_high']*100:.2f}]")

    with open('results/full_condition_summary_v1_corrected.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'name', 'mean', 'sd', 'hdi94_low', 'hdi94_high'])
        w.writerows(rows)
    print("Saved: results/full_condition_summary_v1_corrected.csv")


if __name__ == '__main__':
    main()
