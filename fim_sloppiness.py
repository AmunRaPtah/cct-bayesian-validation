"""
Structural identifiability / "sloppy model" analysis for the v2 (3-drug)
calibration.

Reviewer concern: with 14 free parameters and only 1-3 real behavioral
anchor points, most parameter combinations cannot be locally identified by
the data; the model is a candidate for what Gutenkunst et al. (2007,
PLoS Comp Bio, "Universally sloppy parameter sensitivities in systems
biology") call a "sloppy model" -- a small number of stiff parameter
combinations are tightly constrained, while an exponentially larger set of
sloppy combinations are essentially unconstrained.

Method: the v2 likelihood is built from exactly 3 real numbers
(R_ARI, R_MEM, R_BUS; bayesian_calibration.py). We compute the Jacobian
J = d(R_ARI, R_MEM, R_BUS) / d(theta_1..theta_14) by central finite
differences at a representative parameter point, form the Fisher Information
Matrix FIM = J^T diag(1/sigma^2) J (14x14, rank <= 3 by construction), and
eigendecompose it. The eigenvalue spectrum directly answers "which parameter
combinations are actually constrained by the data": eigenvalues near zero
correspond to sloppy (unconstrained) directions; the top few eigenvalues and
their eigenvectors show the stiff combinations the MEM/ARI/BUS anchors do
constrain.
"""
import json
import sys

import numpy as np

from priors import PRIOR_SPECS, FREE_PARAM_NAMES
from evidence import ARI_CPP_EVIDENCE, MEM_CONSOLIDATION_EVIDENCE, BUS_CPP_EVIDENCE
from bayesian_calibration import _simulate_R_vector_raw

SIGMAS = np.array([
    ARI_CPP_EVIDENCE['R_ARI_target_sd'],
    MEM_CONSOLIDATION_EVIDENCE['R_MEM_target_sd'],
    BUS_CPP_EVIDENCE['R_BUS_target_sd'],
])


def literature_midpoint_vec():
    vec = []
    for name in FREE_PARAM_NAMES:
        spec = PRIOR_SPECS[name]
        lo, hi = spec[-2], spec[-1]
        vec.append((lo + hi) / 2)
    return np.array(vec)


def jacobian(vec, rel_step=1e-3):
    n = len(vec)
    J = np.zeros((3, n))
    for j in range(n):
        h = max(abs(vec[j]) * rel_step, 1e-5)
        v_plus, v_minus = vec.copy(), vec.copy()
        v_plus[j] += h
        v_minus[j] -= h
        R_plus = _simulate_R_vector_raw(v_plus)
        R_minus = _simulate_R_vector_raw(v_minus)
        J[:, j] = (R_plus - R_minus) / (2 * h)
    return J


def analyze(vec, label):
    J = jacobian(vec)
    J_weighted = J / SIGMAS[:, None]
    FIM = J_weighted.T @ J_weighted  # 14x14, rank <= 3

    eigvals, eigvecs = np.linalg.eigh(FIM)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_clipped = np.clip(eigvals, 1e-12, None)
    log_eigvals = np.log10(eigvals_clipped)
    spread = float(log_eigvals[0] - log_eigvals[-1])

    top_combos = []
    for k in range(min(3, len(eigvals))):
        vec_k = eigvecs[:, k]
        loadings = sorted(zip(FREE_PARAM_NAMES, vec_k), key=lambda t: -abs(t[1]))
        top_combos.append({
            'eigenvalue': float(eigvals[k]),
            'log10_eigenvalue': float(log_eigvals[k]),
            'top_loadings': [(n, float(w)) for n, w in loadings[:5]],
        })

    n_sloppy = int(np.sum(eigvals < 1e-6 * eigvals[0]))

    result = {
        'label': label,
        'param_point': {n: float(v) for n, v in zip(FREE_PARAM_NAMES, vec)},
        'eigenvalues_desc': [float(v) for v in eigvals],
        'log10_eigenvalues_desc': [float(v) for v in log_eigvals],
        'log_spread_max_over_min': spread,
        'n_params': len(vec),
        'n_output_dims_in_likelihood': 3,
        'n_effectively_unconstrained_directions': n_sloppy,
        'stiffest_combinations': top_combos,
    }
    return result


def main():
    vec = literature_midpoint_vec()
    result = analyze(vec, 'literature_midpoint')
    with open('results/fim_sloppiness_v2.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
