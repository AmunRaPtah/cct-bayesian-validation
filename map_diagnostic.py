"""
Independent trust-region MAP-finding diagnostic for the v2 (3-drug) Bayesian
calibration, to be checked against the poorly-converged DEMetropolisZ
posterior (max r-hat 1.35, min bulk ESS 10; see bayesian_calibration.py /
results/calibration_diagnostics_v2.txt).

This is exactly the diagnostic the paper's own Section 5.5 names as planned
but unresolved: "a trust-region point-estimate approach such as Fides
[Froehlich and Sorger 2022] as an independent MAP-finding diagnostic against
which the MCMC posterior can be checked." We don't have Fides installed, but
scipy.optimize.least_squares(method='trf') is the same class of algorithm
(trust-region-reflective) and needs no gradient of the ODE solve -- it builds
its own finite-difference Jacobian, exactly like the black-box treatment
DEMetropolisZ already assumes.

Approach: minimize the same v2 log-posterior (Gaussian likelihood on
R_ARI/R_MEM/R_BUS + Gaussian log-prior terms for the elicited/truncnorm
parameters) formulated as a nonlinear least-squares problem, from many random
box-constrained starts, and keep the best local optimum. Uniform priors are
enforced via the optimizer's own bounds; truncnorm priors get an explicit
z-score residual term. Report the MAP super-additivity index against the
MCMC v2 posterior mean (21.9 pp) and v1's much better-converged 13.0 pp.
"""
import json
import time

import numpy as np
from scipy.optimize import least_squares

from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES
from evidence import ARI_CPP_EVIDENCE, MEM_CONSOLIDATION_EVIDENCE, BUS_CPP_EVIDENCE
import cct_model as cm
from bayesian_calibration import _simulate_R_vector_raw, build_param_dict

SEED = 20260726
LOWER = np.array([PRIOR_SPECS[n][-2] for n in FREE_PARAM_NAMES])
UPPER = np.array([PRIOR_SPECS[n][-1] for n in FREE_PARAM_NAMES])

TARGETS = np.array([
    ARI_CPP_EVIDENCE['R_ARI_target_mean'],
    MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_mean'],
    BUS_CPP_EVIDENCE['R_BUS_target_mean'],
])
SIGMAS = np.array([
    ARI_CPP_EVIDENCE['R_ARI_target_sd'],
    MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_sd'],
    BUS_CPP_EVIDENCE['R_BUS_target_sd'],
])

# Which free params carry an elicited (truncnorm) prior -> get a prior residual
TRUNCNORM_PARAMS = {name: spec for name, spec in PRIOR_SPECS.items() if spec[0] == 'truncnorm'}


def residuals(vec):
    R_sim = _simulate_R_vector_raw(vec)
    if np.any(R_sim <= -998):
        return np.full(3 + len(TRUNCNORM_PARAMS), 1e3)
    lik_res = (R_sim - TARGETS) / SIGMAS
    prior_res = []
    for name, spec in TRUNCNORM_PARAMS.items():
        _, mean, sd, lo, hi = spec
        val = vec[FREE_PARAM_NAMES.index(name)]
        prior_res.append((val - mean) / sd)
    return np.concatenate([lik_res, np.array(prior_res)])


def run(n_restarts=25):
    rng = np.random.default_rng(SEED)
    best = None
    t0 = time.time()
    for i in range(n_restarts):
        x0 = rng.uniform(LOWER, UPPER)
        try:
            sol = least_squares(residuals, x0, bounds=(LOWER, UPPER), method='trf',
                                 xtol=1e-10, ftol=1e-10)
        except Exception as e:
            print(f'restart {i}: failed ({e})')
            continue
        cost = float(sol.cost)
        if best is None or cost < best['cost']:
            best = {'cost': cost, 'x': sol.x.copy(), 'restart': i}
        print(f'restart {i}: cost={cost:.5f}')
    t1 = time.time()
    print(f'Elapsed: {t1 - t0:.1f}s over {n_restarts} restarts')

    map_vec = best['x']
    map_params = build_param_dict(map_vec)
    R_sim = _simulate_R_vector_raw(map_vec)

    factorial = cm.run_factorial(map_params)
    sa = cm.compute_super_additivity(factorial)

    result = {
        'best_cost': best['cost'],
        'best_restart': best['restart'],
        'n_restarts': n_restarts,
        'map_params': {n: float(v) for n, v in zip(FREE_PARAM_NAMES, map_vec)},
        'R_sim_at_MAP': {'R_ARI': float(R_sim[0]), 'R_MEM': float(R_sim[1]), 'R_BUS': float(R_sim[2])},
        'R_targets': {'R_ARI': float(TARGETS[0]), 'R_MEM': float(TARGETS[1]), 'R_BUS': float(TARGETS[2])},
        'super_additivity_index_pp_at_MAP': float(sa['super_additivity_index'] * 100),
        'E_ctrl_at_MAP': float(sa['E_ctrl']),
        'E_combo_at_MAP': float(sa['E_combo']),
    }
    with open('results/map_diagnostic_v2.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run()
