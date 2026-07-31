"""
Reconstructed v1 (MEM-only) Bayesian calibration, run with substantially
longer chains than the original manuscript run.

The original v1 script (bayesian_calibration.py at commit 8281f26, before
the "Extend Bayesian calibration to all three drugs" commit 3b2914e) used
4 chains x 1200 tune x 1200 draws and reported max r_hat = 1.05, min bulk
ESS = 120 -- above the conventional r_hat < 1.01 threshold. That script
imported IDENTIFIED_PARAM_NAMES / UNIDENTIFIED_BY_MEM_LIKELIHOOD from an
older priors.py that no longer exists (priors.py was refactored when the
v2 three-drug extension replaced it). PRIOR_SPECS themselves are unchanged
between the two versions (verified against git history), so this script
reconstructs the v1 identification logic against the current priors.py
and reruns with longer chains, targeting r_hat < 1.01 and ESS > 200.

phi1, phi3, delta_BUS are analytically unidentified by the MEM-only
likelihood (vehicle and MEM-alone conditions both have ARI=BUS=0), exactly
as in the original v1 script. They are excluded from the MCMC and fixed at
their prior mean/midpoint for the likelihood evaluation.

The paper reports two independent runs of this script with different seeds
to confirm the corrected result is not itself a single-run artifact:
    python bayesian_calibration_v1_longrun.py --seed 20260712 --suffix longrun
    python bayesian_calibration_v1_longrun.py --seed 20260713 --suffix confirm
"""

import argparse
import time
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

from priors import PRIOR_SPECS, STRUCTURAL
from evidence import MEM_CONSOLIDATION_EVIDENCE
import cct_model as cm

VEHICLE_DOSES = [0.0, 0.0, 0.0]
MEM_ALONE_DOSES = [0.0, 1.0, 0.0]

UNIDENTIFIED_BY_MEM_LIKELIHOOD = ['phi1', 'phi3', 'delta_BUS']
FREE_PARAM_NAMES = list(PRIOR_SPECS.keys())
IDENTIFIED_PARAM_NAMES = [n for n in FREE_PARAM_NAMES if n not in UNIDENTIFIED_BY_MEM_LIKELIHOOD]

_FIXED_UNIDENTIFIED = {}
for _name in UNIDENTIFIED_BY_MEM_LIKELIHOOD:
    _spec = PRIOR_SPECS[_name]
    _FIXED_UNIDENTIFIED[_name] = _spec[1] if _spec[0] == 'truncnorm' else (_spec[1] + _spec[2]) / 2


def build_param_dict(vec):
    p = dict(STRUCTURAL)
    p.update(_FIXED_UNIDENTIFIED)
    for name, val in zip(IDENTIFIED_PARAM_NAMES, vec):
        p[name] = float(val)
    return p


def _simulate_R_MEM_raw(vec):
    p = build_param_dict(vec)
    try:
        E_ctrl = cm.final_E(VEHICLE_DOSES, p)
        E_mem = cm.final_E(MEM_ALONE_DOSES, p)
    except RuntimeError:
        return np.array(-999.0)
    if E_ctrl <= 1e-6:
        return np.array(-999.0)
    return np.array((E_ctrl - E_mem) / E_ctrl)


@as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
def simulate_R_MEM(vec):
    return _simulate_R_MEM_raw(vec)


def build_model():
    with pm.Model() as model:
        rvs = []
        for name in IDENTIFIED_PARAM_NAMES:
            spec = PRIOR_SPECS[name]
            if spec[0] == 'uniform':
                _, lo, hi = spec
                rv = pm.Uniform(name, lower=lo, upper=hi)
            else:
                _, mean, sd, lo, hi = spec
                rv = pm.TruncatedNormal(name, mu=mean, sigma=sd, lower=lo, upper=hi)
            rvs.append(rv)
        param_vec = pt.stack(rvs)
        R_MEM_sim = pm.Deterministic('R_MEM_sim', simulate_R_MEM(param_vec))
        pm.Normal(
            'R_MEM_obs',
            mu=R_MEM_sim,
            sigma=MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_sd'],
            observed=MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_mean'],
        )
    return model


def main(seed, suffix, chains=6, tune=4000, draws=4000):
    model = build_model()
    t0 = time.time()

    with model:
        prior_idata = pm.sample_prior_predictive(draws=2000, random_seed=seed)
        step = pm.DEMetropolisZ()
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, step=step,
            random_seed=seed, progressbar=False,
        )

    summary = az.summary(idata, var_names=IDENTIFIED_PARAM_NAMES + ['R_MEM_sim'])
    print(summary.to_string())
    summary.to_csv(f'results/posterior_summary_v1_{suffix}.csv')

    r_hat_max = float(pd.to_numeric(summary['r_hat'], errors='coerce').max())
    ess_min = float(pd.to_numeric(summary['ess_bulk'], errors='coerce').min())
    elapsed = time.time() - t0
    print(f"\nMax r_hat: {r_hat_max:.4f}  (target < 1.01)")
    print(f"Min ESS (bulk): {ess_min:.0f}  (target > 200)")
    print(f"Elapsed: {elapsed:.0f}s  seed={seed} chains={chains} tune={tune} draws={draws}")

    with open(f'results/calibration_diagnostics_v1_{suffix}.txt', 'w') as f:
        f.write(f"seed={seed}\nmax_r_hat={r_hat_max}\nmin_ess_bulk={ess_min}\n"
                f"chains={chains}\ntune={tune}\ndraws={draws}\nelapsed_s={elapsed}\n")

    idata.to_netcdf(f'results/cct_posterior_v1_{suffix}.nc')
    prior_idata.to_netcdf(f'results/cct_prior_v1_{suffix}.nc')
    print(f"Saved: results/cct_posterior_v1_{suffix}.nc, results/cct_prior_v1_{suffix}.nc")

    return idata, prior_idata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260712)
    parser.add_argument('--suffix', type=str, default='longrun',
                         help="output file suffix, e.g. 'longrun' or 'confirm'")
    parser.add_argument('--chains', type=int, default=6)
    parser.add_argument('--tune', type=int, default=4000)
    parser.add_argument('--draws', type=int, default=4000)
    args = parser.parse_args()
    main(args.seed, args.suffix, args.chains, args.tune, args.draws)
