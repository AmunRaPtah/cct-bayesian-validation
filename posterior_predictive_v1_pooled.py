"""
Final corrected v1 (MEM-only) posterior-predictive super-additivity
distribution, pooling both independent long-chain runs
(cct_posterior_v1_longrun.nc, seed 20260712; cct_posterior_v1_confirm.nc,
seed 20260713) into a single n=1600 draw sample.

Both runs converged to essentially the same posterior (see
results/posterior_summary_v1_longrun.csv vs _confirm.csv) and both closely
match the calibration target (R_MEM_sim ~ 0.60, the model's own likelihood
mean), unlike the original short run reported in the manuscript
(R_MEM_sim = 0.785, ~3.7 SD from its own calibration target despite passing
r_hat/ESS diagnostics -- a convergence artifact this rerun corrects).

phi1, phi3, delta_BUS are not in the v1 trace (analytically unidentified by
the MEM-only likelihood) and are backfilled by independent resampling from
their own literature-range priors per draw, exactly as
posterior_predictive_full.py does.
"""
import numpy as np
import arviz as az

from priors import STRUCTURAL, IDENTIFIED_PARAM_NAMES, scipy_frozen_dists
import cct_model as cm

SEED = 20260714
N_DRAWS_PER_RUN = 800

_PRIOR_DISTS = scipy_frozen_dists()


def param_dict_from_row(row, rng, missing_names):
    p = dict(STRUCTURAL)
    p.update(row)
    for name in missing_names:
        p[name] = float(_PRIOR_DISTS[name].rvs(random_state=rng))
    return p


def draws_from_trace(idata_path, n_draws, rng):
    idata = az.from_netcdf(idata_path)
    ds = idata['posterior'].to_dataset()
    available = set(ds.data_vars)
    present_names = [n for n in IDENTIFIED_PARAM_NAMES if n in available]
    missing_names = [n for n in IDENTIFIED_PARAM_NAMES if n not in available]
    flat = ds.stack(sample=("chain", "draw"))
    n_total = flat.sizes["sample"]
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

    sa_vals = []
    failures = 0
    for i in idx:
        row = {name: float(flat[name].values[i]) for name in present_names}
        p = param_dict_from_row(row, rng, missing_names)
        try:
            results = cm.run_factorial(p)
            sa = cm.compute_super_additivity(results)
        except RuntimeError:
            failures += 1
            continue
        sa_vals.append(sa['super_additivity_index'])
    return np.array(sa_vals), failures


def summarize(arr):
    hdi = az.hdi(arr, prob=0.94)
    return {
        'n': len(arr), 'mean': float(arr.mean()), 'sd': float(arr.std()),
        'hdi_low': float(hdi[0]), 'hdi_high': float(hdi[1]),
        'p_positive': float(np.mean(arr > 0)),
    }


def main():
    rng = np.random.default_rng(SEED)
    a, fail_a = draws_from_trace('results/cct_posterior_v1_longrun.nc', N_DRAWS_PER_RUN, rng)
    b, fail_b = draws_from_trace('results/cct_posterior_v1_confirm.nc', N_DRAWS_PER_RUN, rng)
    pooled = np.concatenate([a, b])

    print(f"Run 1 (seed 20260712): n={len(a)} failures={fail_a}")
    print(f"  {summarize(a)}")
    print(f"Run 2 (seed 20260713): n={len(b)} failures={fail_b}")
    print(f"  {summarize(b)}")
    print(f"Pooled (n={len(pooled)}):")
    s = summarize(pooled)
    print(f"  {s}")
    print(f"\nSuper-additivity index: {s['mean']*100:.1f} pp "
          f"(94% HDI [{s['hdi_low']*100:.1f}, {s['hdi_high']*100:.1f}]), "
          f"P(index>0)={s['p_positive']:.4f}, n={s['n']}")

    import json
    with open('results/super_additivity_v1_corrected_pooled.json', 'w') as f:
        json.dump({'run1': summarize(a), 'run2': summarize(b), 'pooled': s}, f, indent=2)
    print("\nSaved: results/super_additivity_v1_corrected_pooled.json")


if __name__ == '__main__':
    main()
