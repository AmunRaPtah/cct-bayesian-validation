"""
Robustness check: does the model's qualitative prediction survive if MEM's
amplitude-suppression channel (phi2) contributes nothing at all?

Motivated by Ma, Mufti, and Leung (2015): memantine enhanced, rather than
suppressed, hippocampal CA1 LTP in freely-moving rats. The paper (Section
6.1) already reports this as an open discrepancy for a naive "memantine
suppresses Axis-2 amplitude" reading. This script asks a narrower, testable
question: with phi2 pinned at exactly 0 (no MEM amplitude-suppression effect
at all, the most literal reading of Ma et al.'s finding transplanted onto
Axis 2), does the model's *threshold*-only mechanism (delta_MEM, acting on
the encoding-probability threshold rather than on A2 amplitude) still
predict (a) a reduction in R_MEM and (b) a positive super-additivity index
for the triple combination, across the full literature-range uncertainty in
the other 13 parameters?

This does not resolve the CA1-vs-NAc regional/subunit discrepancy -- that
requires new data -- but it does show whether the model's core claims are
contingent on the amplitude-suppression channel specifically, or survive on
the threshold-elevation channel alone.
"""
import json
import time
from multiprocessing import Pool

import numpy as np

from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES
import cct_model as cm

N = 2000
SEED = 20260726
NPROC = 2

OTHER_PARAMS = [n for n in FREE_PARAM_NAMES if n != 'phi2']


def sample_others(rng):
    p = {}
    for name in OTHER_PARAMS:
        spec = PRIOR_SPECS[name]
        if spec[0] == 'uniform':
            _, lo, hi = spec
            p[name] = rng.uniform(lo, hi)
        else:
            _, mean, sd, lo, hi = spec
            # rejection sample within literature bounds
            while True:
                v = rng.normal(mean, sd)
                if lo <= v <= hi:
                    p[name] = v
                    break
    return p


def eval_one(seed):
    rng = np.random.default_rng(seed)
    p = dict(STRUCTURAL)
    p.update(sample_others(rng))
    p['phi2'] = 0.0
    try:
        E_ctrl = cm.final_E([0.0, 0.0, 0.0], p)
        E_mem = cm.final_E([0.0, 1.0, 0.0], p)
        res = cm.run_factorial(p)
        sa = cm.compute_super_additivity(res)
        R_MEM = (E_ctrl - E_mem) / E_ctrl if E_ctrl > 1e-6 else np.nan
        return R_MEM, sa['super_additivity_index']
    except Exception:
        return np.nan, np.nan


def main():
    seeds = list(range(SEED, SEED + N))
    t0 = time.time()
    with Pool(NPROC) as pool:
        results = pool.map(eval_one, seeds, chunksize=20)
    t1 = time.time()
    R_MEM = np.array([r[0] for r in results])
    SA = np.array([r[1] for r in results])

    valid = ~np.isnan(R_MEM)
    summary = {
        'N': N,
        'n_valid': int(valid.sum()),
        'elapsed_s': t1 - t0,
        'phi2_fixed_at': 0.0,
        'R_MEM': {
            'frac_positive': float(np.mean(R_MEM[valid] > 0)),
            'mean': float(np.mean(R_MEM[valid])),
            'min': float(np.min(R_MEM[valid])),
            'max': float(np.max(R_MEM[valid])),
            'pct_2_5': float(np.percentile(R_MEM[valid], 2.5)),
            'pct_97_5': float(np.percentile(R_MEM[valid], 97.5)),
        },
        'super_additivity_index_pp': {
            'frac_positive': float(np.mean(SA[valid] > 0)),
            'mean_pp': float(np.mean(SA[valid]) * 100),
            'min_pp': float(np.min(SA[valid]) * 100),
            'max_pp': float(np.max(SA[valid]) * 100),
            'pct_2_5_pp': float(np.percentile(SA[valid], 2.5) * 100),
            'pct_97_5_pp': float(np.percentile(SA[valid], 97.5) * 100),
        },
    }
    with open('results/phi2_zero_robustness.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
