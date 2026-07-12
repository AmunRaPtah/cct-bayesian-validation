"""
Posterior-predictive validation of the CCT model's central claim: that
simultaneous sub-threshold suppression of all three axes (ARI+MEM+BUS)
produces a super-additive (non-independent) reduction in encoding
probability, beyond the Bliss-independence expectation from the three
single-agent effects.

This is the "we can't do wet-lab experiments" substitute: instead of a
single point-estimate simulation (as in the public reference code), we
propagate full posterior parameter uncertainty (from bayesian_calibration.py)
through the 2x2x2 factorial ODE simulation and report a *distribution* of
the super-additivity index, with a 94% credible interval and P(index > 0).

We report this both for:
  - the PRIOR-predictive (before conditioning on the real MEM behavioral
    datum) -- i.e. "what does the architecture predict given literature-range
    uncertainty alone"
  - the POSTERIOR-predictive (after conditioning) -- "what does it predict
    once we fold in the one real quantitative behavioral anchor we have"
so the effect of the Bayesian update itself is visible, not just the final
number.
"""

import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from priors import STRUCTURAL, IDENTIFIED_PARAM_NAMES, UNIDENTIFIED_BY_MEM_LIKELIHOOD, scipy_frozen_dists
import cct_model as cm

SEED = 20260712
N_POSTERIOR_DRAWS = 800  # subsample for the 8-condition factorial (8x ODE solves each)

# phi1, phi3, delta_BUS were not part of the MCMC (unidentified by the MEM
# likelihood -- see priors.py). Here, where ARI/BUS doses ARE exercised, we
# reintroduce them by drawing independently from their own literature-range
# priors and merging with each (posterior or prior) draw of the identified
# parameters. This keeps them at "prior-only" status, exactly as the
# evidence supports.
_UNIDENTIFIED_DISTS = {name: scipy_frozen_dists()[name] for name in UNIDENTIFIED_BY_MEM_LIKELIHOOD}


def param_dict_from_row(row, rng):
    p = dict(STRUCTURAL)
    for name in IDENTIFIED_PARAM_NAMES:
        p[name] = float(row[name])
    for name, dist in _UNIDENTIFIED_DISTS.items():
        p[name] = float(dist.rvs(random_state=rng))
    return p


def run_super_additivity_distribution(idata_path, group, n_draws, rng):
    idata = az.from_netcdf(idata_path)
    ds = idata[group].to_dataset()
    flat = ds.stack(sample=("chain", "draw"))
    n_total = flat.sizes["sample"]
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

    sa_vals, r_combo_vals, r_indep_vals = [], [], []
    failures = 0
    for i in idx:
        row = {name: float(flat[name].values[i]) for name in IDENTIFIED_PARAM_NAMES}
        p = param_dict_from_row(row, rng)
        try:
            results = cm.run_factorial(p)
            sa = cm.compute_super_additivity(results)
        except RuntimeError:
            failures += 1
            continue
        sa_vals.append(sa['super_additivity_index'])
        r_combo_vals.append(sa['R_combo'])
        r_indep_vals.append(sa['R_indep'])

    return {
        'super_additivity_index': np.array(sa_vals),
        'R_combo': np.array(r_combo_vals),
        'R_indep': np.array(r_indep_vals),
        'n_failures': failures,
        'n_used': len(sa_vals),
    }


def summarize(name, arr):
    hdi = az.hdi(arr, prob=0.94)
    p_positive = float(np.mean(arr > 0))
    print(f"\n--- {name} ---")
    print(f"  n = {len(arr)}")
    print(f"  mean = {arr.mean():.4f}")
    print(f"  sd   = {arr.std():.4f}")
    print(f"  94% HDI = [{hdi[0]:.4f}, {hdi[1]:.4f}]")
    print(f"  P(super-additivity index > 0) = {p_positive:.4f}")
    return {
        'mean': float(arr.mean()), 'sd': float(arr.std()),
        'hdi_low': float(hdi[0]), 'hdi_high': float(hdi[1]),
        'p_positive': p_positive, 'n': len(arr),
    }


def main():
    rng = np.random.default_rng(SEED)

    print("Running PRIOR-predictive factorial simulation "
          f"({N_POSTERIOR_DRAWS} draws from literature-range priors)...")
    prior_res = run_super_additivity_distribution(
        'results/cct_prior.nc', 'prior', N_POSTERIOR_DRAWS, rng)
    prior_summary = summarize("PRIOR-predictive super-additivity index (pp, as fraction)",
                               prior_res['super_additivity_index'])

    print("\nRunning POSTERIOR-predictive factorial simulation "
          f"({N_POSTERIOR_DRAWS} draws, conditioned on real MEM behavioral datum)...")
    post_res = run_super_additivity_distribution(
        'results/cct_posterior.nc', 'posterior', N_POSTERIOR_DRAWS, rng)
    post_summary = summarize("POSTERIOR-predictive super-additivity index (pp, as fraction)",
                              post_res['super_additivity_index'])

    # --- figure: prior vs posterior predictive distributions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(prior_res['super_additivity_index'] * 100, bins=40, alpha=0.5,
            label=f"Prior-predictive (n={prior_res['n_used']})", color='#7fa8d9')
    ax.hist(post_res['super_additivity_index'] * 100, bins=40, alpha=0.6,
            label=f"Posterior-predictive (n={post_res['n_used']})", color='#d97f7f')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Super-additivity index (percentage points, R_combo - R_indep)')
    ax.set_ylabel('Count (posterior/prior draws)')
    ax.set_title('CCT model: posterior-predictive validation of super-additivity\n'
                  '(in silico substitute for wet-lab replication)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('results/super_additivity_distribution.png', dpi=180)
    plt.close()
    print("\nSaved: results/super_additivity_distribution.png")

    import csv
    with open('results/super_additivity_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distribution', 'mean', 'sd', 'hdi94_low', 'hdi94_high', 'p_positive', 'n'])
        w.writerow(['prior_predictive', prior_summary['mean'], prior_summary['sd'],
                    prior_summary['hdi_low'], prior_summary['hdi_high'],
                    prior_summary['p_positive'], prior_summary['n']])
        w.writerow(['posterior_predictive', post_summary['mean'], post_summary['sd'],
                    post_summary['hdi_low'], post_summary['hdi_high'],
                    post_summary['p_positive'], post_summary['n']])
    print("Saved: results/super_additivity_summary.csv")

    return prior_summary, post_summary


if __name__ == '__main__':
    main()
