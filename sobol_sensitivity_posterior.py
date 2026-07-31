"""
Global (Sobol) sensitivity analysis restricted to the v1 posterior, rather
than the full literature-range prior (sobol_sensitivity.py).

Reviewer concern: sobol_sensitivity.py's ranking (lambda1 and r_E dominating
total-order variance, gate parameters near zero) is computed over the full
Table 1 literature ranges, i.e. under the PRIOR. Since lambda1 has a widened
prior and is not tightly identified by the v1 (MEM-only) likelihood, the
dominance of lambda1/r_E "may reflect prior specification rather than
biological sensitivity" (reviewer's words). This script recomputes the same
Sobol decomposition using each parameter's 94% HDI from the v1 posterior
(cct_posterior_v1_longrun.nc, produced by bayesian_calibration_v1_longrun.py)
in place of its literature-range bound, wherever the v1 likelihood actually
identifies that parameter.

Caveats stated explicitly in the output:
  1. This treats each parameter's marginal posterior range independently
     (uniform over its 94% HDI) for Saltelli sampling, ignoring posterior
     correlations between parameters. A fully joint posterior-based Sobol
     decomposition would require a different sampling scheme (e.g.
     resampling directly from the joint posterior trace); this is a
     standard, declared approximation, not the exact posterior variance
     decomposition.
  2. phi1, phi3, delta_BUS are analytically unidentified by the v1 (MEM-only)
     likelihood (priors.py) and are absent from the v1 posterior trace
     entirely -- their bounds here remain the full literature-range prior,
     unchanged from sobol_sensitivity.py, and this is stated in the output
     rather than silently narrowed.
"""
import sys
import os
import json
import time
from multiprocessing import Pool

import numpy as np
import arviz as az
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(__file__))
import cct_model as cm
from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES

UNIDENTIFIED_BY_V1 = ['phi1', 'phi3', 'delta_BUS']


def literature_bounds(name):
    spec = PRIOR_SPECS[name]
    if spec[0] == 'uniform':
        return spec[1], spec[2]
    return spec[3], spec[4]


def posterior_hdi_bounds(idata_path, prob=0.94):
    idata = az.from_netcdf(idata_path)
    posterior = idata['posterior']
    bounds = {}
    for name in FREE_PARAM_NAMES:
        if name in UNIDENTIFIED_BY_V1:
            continue
        if name not in posterior.data_vars:
            continue
        hdi = az.hdi(posterior[name], prob=prob)
        lo = float(hdi.sel(ci_bound='lower').values)
        hi = float(hdi.sel(ci_bound='upper').values)
        bounds[name] = (lo, hi)
    return bounds


def build_problem(idata_path):
    post_bounds = posterior_hdi_bounds(idata_path)
    bounds = []
    source = {}
    for name in FREE_PARAM_NAMES:
        if name in post_bounds:
            lo, hi = post_bounds[name]
            source[name] = 'posterior_94hdi'
        else:
            lo, hi = literature_bounds(name)
            source[name] = 'literature_range_prior (unidentified by v1)'
        bounds.append([lo, hi])
    problem = {'num_vars': len(FREE_PARAM_NAMES), 'names': FREE_PARAM_NAMES, 'bounds': bounds}
    return problem, source


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
    idata_path = sys.argv[1] if len(sys.argv) > 1 else 'results/cct_posterior_v1_longrun.nc'
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    problem, source = build_problem(idata_path)
    print('Parameter bounds used (source):', flush=True)
    for name, (lo, hi) in zip(problem['names'], problem['bounds']):
        print(f'  {name}: [{lo:.4f}, {hi:.4f}]  <- {source[name]}', flush=True)

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
        'note': 'Sobol indices computed with parameter ranges drawn from the v1 '
                'posterior 94% HDI where identified, literature-range prior '
                'otherwise (phi1, phi3, delta_BUS). Marginal ranges only, not a '
                'joint posterior resampling; posterior correlations are not '
                'captured. See module docstring.',
        'idata_path': idata_path,
        'bounds_source': source,
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
    with open('results/sobol_sensitivity_posterior_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    np.save('results/sobol_posterior_super_additivity_samples.npy', Y)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
