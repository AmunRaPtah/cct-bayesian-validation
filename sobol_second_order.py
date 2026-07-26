"""
Second-order Sobol index for the gate-parameter pair (gamma_g x theta_g),
requested to test the analytic d2P/dA1dA2 > 0 cross-partial claim in
Section 3.3 against a variance-based interaction measure. Reuses the same
problem definition as sobol_sensitivity.py but with calc_second_order=True,
which is more expensive (N*(2D+2) samples instead of N*(D+2)), so this is
run at a smaller N than the primary total/first-order run.
"""
import sys, os, json, time
from multiprocessing import Pool
import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(__file__))
from sobol_sensitivity import problem, eval_row, FREE_PARAM_NAMES

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    param_values = sobol_sample.sample(problem, N, calc_second_order=True)
    print(f'N={N}, total evaluations: {len(param_values)}', flush=True)
    t0 = time.time()
    with Pool(nproc) as pool:
        Y = np.array(pool.map(eval_row, param_values, chunksize=20))
    t1 = time.time()
    print(f'Elapsed: {t1-t0:.1f}s', flush=True)
    np.save('results/sobol_second_order_Y.npy', Y)  # cache raw output before analysis, in case of a bug downstream
    n_nan = int(np.isnan(Y).sum())
    Y_clean = Y.copy()
    if n_nan:
        Y_clean[np.isnan(Y_clean)] = np.nanmedian(Y_clean)
    Si = sobol_analyze.analyze(problem, Y_clean, calc_second_order=True, print_to_console=False)
    gi, ti = FREE_PARAM_NAMES.index('gamma_g'), FREE_PARAM_NAMES.index('theta_g')
    # SALib's S2 matrix is only populated for row < col (upper triangle); take
    # whichever of the two orderings is not NaN.
    lo, hi = min(gi, ti), max(gi, ti)
    s2_gate = float(Si['S2'][lo, hi])
    s2_gate_conf = float(Si['S2_conf'][lo, hi])
    result = {
        'N': N, 'total_evaluations': len(param_values), 'n_failed': n_nan,
        'S2_gamma_g_theta_g': s2_gate, 'S2_conf_gamma_g_theta_g': s2_gate_conf,
        'frac_positive': float(np.mean(Y > 0)),
    }
    with open('results/sobol_second_order_summary.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
