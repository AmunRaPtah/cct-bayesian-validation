"""
Global (Sobol) sensitivity analysis over all 14 free CCT parameters.

Addresses a reviewer critique of the deterministic 6x6 grid in paper Section
5.4: that grid varies only gamma_g and theta_g, holding the other 12
parameters at fixed literature midpoints, so it cannot show whether the
super-additivity direction is an architectural property or an artifact of
those midpoint choices. This script runs Saltelli sampling across all 14
parameters' full literature-range bounds (Table 1 / priors.py) and computes
first-order (S1) and total-order (ST) Sobol indices for the super-additivity
index, plus the fraction of samples with a positive index across the entire
14-dimensional space.

Independent of the MCMC pipeline (bayesian_calibration.py): no PyMC, no
convergence diagnostics to worry about, since this is plain forward
Monte Carlo / variance decomposition over the deterministic ODE simulator.
"""
import sys
import os
import json
import time
from multiprocessing import Pool

import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(__file__))
import cct_model as cm
from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES

BOUNDS = []
for name in FREE_PARAM_NAMES:
    spec = PRIOR_SPECS[name]
    if spec[0] == 'uniform':
        lo, hi = spec[1], spec[2]
    else:  # truncnorm -- use the literature bounds (Table 1), not mean/sd
        lo, hi = spec[3], spec[4]
    BOUNDS.append([lo, hi])

problem = {
    'num_vars': len(FREE_PARAM_NAMES),
    'names': FREE_PARAM_NAMES,
    'bounds': BOUNDS,
}


def eval_row(row):
    p = dict(STRUCTURAL)
    p.update(dict(zip(FREE_PARAM_NAMES, row)))
    try:
        res = cm.run_factorial(p)
        sa = cm.compute_super_additivity(res)
        return sa['super_additivity_index']
    except Exception:
        return np.nan


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    param_values = sobol_sample.sample(problem, N, calc_second_order=False)
    print(f'N={N}, total evaluations: {len(param_values)}', flush=True)

    t0 = time.time()
    with Pool(nproc) as pool:
        Y = np.array(pool.map(eval_row, param_values, chunksize=20))
    t1 = time.time()
    print(f'Elapsed: {t1 - t0:.1f}s', flush=True)

    n_nan = int(np.isnan(Y).sum())
    print(f'Failed ODE evaluations: {n_nan}/{len(Y)}', flush=True)
    Y_clean = Y.copy()
    if n_nan:
        Y_clean[np.isnan(Y_clean)] = np.nanmedian(Y_clean)

    Si = sobol_analyze.analyze(problem, Y_clean, calc_second_order=False, print_to_console=False)

    summary = {
        'N': N,
        'total_evaluations': len(param_values),
        'n_failed': n_nan,
        'frac_positive_super_additivity': float(np.mean(Y > 0)),
        'min_index_pp': float(np.nanmin(Y) * 100),
        'max_index_pp': float(np.nanmax(Y) * 100),
        'mean_index_pp': float(np.nanmean(Y) * 100),
        'median_index_pp': float(np.nanmedian(Y) * 100),
        'S1': {name: float(v) for name, v in zip(FREE_PARAM_NAMES, Si['S1'])},
        'S1_conf': {name: float(v) for name, v in zip(FREE_PARAM_NAMES, Si['S1_conf'])},
        'ST': {name: float(v) for name, v in zip(FREE_PARAM_NAMES, Si['ST'])},
        'ST_conf': {name: float(v) for name, v in zip(FREE_PARAM_NAMES, Si['ST_conf'])},
    }

    os.makedirs('results', exist_ok=True)
    with open('results/sobol_sensitivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    np.save('results/sobol_super_additivity_samples.npy', Y)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
