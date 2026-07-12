"""
Bayesian calibration of all 14 free CCT model parameters against three real
behavioral single-agent effect sizes -- one per drug/axis -- from the CCT
Master Parameter Table S1 (the project's own authoritative literature-range
table):
  - ARI alone: 40-60% CPP reduction (Wee et al. 2007)          -> identifies phi1 (+ shared lambda1/theta1)
  - MEM alone: 50-70% CPP acquisition block (Popik & Danysz 1997) -> identifies phi2, delta_MEM (+ shared lambda2/theta2_base/theta_g/gamma_g)
  - BUS alone: 30-50% CPP reduction (Haleem et al. 2014)        -> identifies phi3, delta_BUS (+ shared lambda3/theta3)

Each range is treated as an approximate 95% interval (mean=midpoint,
sd=range/4) and used as a Gaussian likelihood on the model's fractional
single-agent encoding-reduction output (R_ARI, R_MEM, R_BUS), computed by
running the ODE forward for that drug alone vs. vehicle. Unlike the earlier
MEM-only version, all 14 parameters are genuinely identified here, since
every drug axis now has its own real behavioral anchor.

The forward model is a black-box ODE solve, not differentiable through
PyTensor, so it's wrapped with `as_op` and sampled with a gradient-free step
method (DEMetropolisZ) rather than NUTS.

Dose-mapping caveat: the real evidence doses fall inside each drug's
manuscript-stated therapeutic range, but the model's dose units are
normalized-to-EC50 (0-1 a.u.) and the true mg/kg-to-EC50 conversion is not
available. We approximate each single-agent evidence dose as a saturating
normalized dose (dose=1.0). This is a stated modeling assumption.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES
from evidence import ARI_CPP_EVIDENCE, MEM_CONSOLIDATION_EVIDENCE, BUS_CPP_EVIDENCE
import cct_model as cm

VEHICLE_DOSES = [0.0, 0.0, 0.0]
ARI_ALONE_DOSES = [1.0, 0.0, 0.0]
MEM_ALONE_DOSES = [0.0, 1.0, 0.0]
BUS_ALONE_DOSES = [0.0, 0.0, 1.0]

SEED = 20260712


def build_param_dict(vec):
    p = dict(STRUCTURAL)
    for name, val in zip(FREE_PARAM_NAMES, vec):
        p[name] = float(val)
    return p


def _simulate_R_vector_raw(vec):
    p = build_param_dict(vec)
    try:
        E_ctrl = cm.final_E(VEHICLE_DOSES, p)
        E_ari = cm.final_E(ARI_ALONE_DOSES, p)
        E_mem = cm.final_E(MEM_ALONE_DOSES, p)
        E_bus = cm.final_E(BUS_ALONE_DOSES, p)
    except RuntimeError:
        return np.array([-999.0, -999.0, -999.0])
    if E_ctrl <= 1e-6:
        return np.array([-999.0, -999.0, -999.0])
    R_ARI = (E_ctrl - E_ari) / E_ctrl
    R_MEM = (E_ctrl - E_mem) / E_ctrl
    R_BUS = (E_ctrl - E_bus) / E_ctrl
    return np.array([R_ARI, R_MEM, R_BUS])


@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def simulate_R_vector(vec):
    return _simulate_R_vector_raw(vec)


def build_model():
    with pm.Model() as model:
        rvs = []
        for name in FREE_PARAM_NAMES:
            spec = PRIOR_SPECS[name]
            if spec[0] == 'uniform':
                _, lo, hi = spec
                rv = pm.Uniform(name, lower=lo, upper=hi)
            else:
                _, mean, sd, lo, hi = spec
                rv = pm.TruncatedNormal(name, mu=mean, sigma=sd, lower=lo, upper=hi)
            rvs.append(rv)
        param_vec = pt.stack(rvs)
        R_sim = pm.Deterministic('R_sim', simulate_R_vector(param_vec))

        pm.Normal('R_ARI_obs', mu=R_sim[0],
                  sigma=ARI_CPP_EVIDENCE['R_ARI_target_sd'],
                  observed=ARI_CPP_EVIDENCE['R_ARI_target_mean'])
        pm.Normal('R_MEM_obs', mu=R_sim[1],
                  sigma=MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_sd'],
                  observed=MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_mean'])
        pm.Normal('R_BUS_obs', mu=R_sim[2],
                  sigma=BUS_CPP_EVIDENCE['R_BUS_target_sd'],
                  observed=BUS_CPP_EVIDENCE['R_BUS_target_mean'])
    return model


def main():
    model = build_model()

    with model:
        prior_idata = pm.sample_prior_predictive(draws=2000, random_seed=SEED)
        step = pm.DEMetropolisZ()
        idata = pm.sample(
            draws=1000, tune=1000, chains=4, cores=1, step=step,
            random_seed=SEED, progressbar=False,
        )

    summary = az.summary(idata, var_names=FREE_PARAM_NAMES)
    print(summary.to_string())
    summary.to_csv('results/posterior_summary_v2.csv')

    r_hat_max = float(pd.to_numeric(summary['r_hat'], errors='coerce').max())
    ess_min = float(pd.to_numeric(summary['ess_bulk'], errors='coerce').min())
    print(f"\nMax r_hat: {r_hat_max:.3f}  (want < 1.05 for good convergence)")
    print(f"Min ESS (bulk): {ess_min:.0f}")

    with open('results/calibration_diagnostics_v2.txt', 'w') as f:
        f.write(f"max_r_hat={r_hat_max}\nmin_ess_bulk={ess_min}\n")

    idata.to_netcdf('results/cct_posterior_v2.nc')
    prior_idata.to_netcdf('results/cct_prior_v2.nc')
    print("Saved: results/cct_posterior_v2.nc, results/cct_prior_v2.nc")

    return idata, prior_idata


if __name__ == '__main__':
    main()
