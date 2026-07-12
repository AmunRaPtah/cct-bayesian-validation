"""
Bayesian calibration of the 11 CCT model parameters that are identified by
the MEM-alone/vehicle likelihood, updated against the one real behavioral
endpoint whose units match the model's output exactly: fractional reduction
in cumulative encoding probability for memantine given alone, during the
consolidation window (evidence.MEM_CONSOLIDATION_EVIDENCE, from Cox/Meyers
et al. 2012, PMC3917839).

phi1, phi3, delta_BUS are excluded from this MCMC (see priors.py) because
ARI=BUS=0 in both conditions used here makes them provably irrelevant to the
likelihood; they keep their literature-range priors and are reintroduced in
posterior_predictive.py, which does exercise ARI/BUS doses.

The forward model (cct_model.run_condition) is a black-box ODE solve, not
differentiable through PyTensor, so it is wrapped with `as_op` and sampled
with a gradient-free step method (DEMetropolisZ) rather than NUTS.

Dose-mapping caveat: the real evidence dose (6.5 mg/kg) falls inside the
manuscript's stated 3-10 mg/kg therapeutic range for memantine, but the
model's dose units are normalized-to-EC50 (0-1 a.u.) and the true
mg/kg-to-EC50 conversion is held privately (patent-pending). We approximate
the evidence dose as a saturating normalized dose (MEM=1.0). This is a
stated modeling assumption, not a measured conversion.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

from priors import PRIOR_SPECS, STRUCTURAL, IDENTIFIED_PARAM_NAMES, UNIDENTIFIED_BY_MEM_LIKELIHOOD
from evidence import MEM_CONSOLIDATION_EVIDENCE
import cct_model as cm

VEHICLE_DOSES = [0.0, 0.0, 0.0]
MEM_ALONE_DOSES = [0.0, 1.0, 0.0]

SEED = 20260712

# phi1, phi3, delta_BUS are unidentified here (see priors.py) -- fixed at
# their prior midpoint for the likelihood evaluation since ARI=BUS=0 makes
# their actual value irrelevant to R_MEM; they are re-sampled from their full
# prior independently in posterior_predictive.py.
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


def main():
    model = build_model()

    with model:
        prior_idata = pm.sample_prior_predictive(draws=2000, random_seed=SEED)
        step = pm.DEMetropolisZ()
        idata = pm.sample(
            draws=1200, tune=1200, chains=4, cores=1, step=step,
            random_seed=SEED, progressbar=False,
        )

    summary = az.summary(idata, var_names=IDENTIFIED_PARAM_NAMES + ['R_MEM_sim'])
    print(summary.to_string())
    summary.to_csv('results/posterior_summary.csv')

    r_hat_max = float(pd.to_numeric(summary['r_hat'], errors='coerce').max())
    ess_min = float(pd.to_numeric(summary['ess_bulk'], errors='coerce').min())
    print(f"\nMax r_hat: {r_hat_max:.3f}  (want < 1.05 for good convergence)")
    print(f"Min ESS (bulk): {ess_min:.0f}")

    with open('results/calibration_diagnostics.txt', 'w') as f:
        f.write(f"max_r_hat={r_hat_max}\nmin_ess_bulk={ess_min}\n")

    idata.to_netcdf('results/cct_posterior.nc')
    prior_idata.to_netcdf('results/cct_prior.nc')
    print("Saved: results/cct_posterior.nc, results/cct_prior.nc")

    return idata, prior_idata


if __name__ == '__main__':
    main()
