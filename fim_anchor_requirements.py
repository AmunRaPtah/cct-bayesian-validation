"""
Minimum-anchor-set analysis for Tier 2 experimental design (reviewer request:
"a table showing how many behavioral anchors of each type -- single,
pairwise, triple -- would be needed to identify all parameters").

Extends fim_sloppiness.py's method (Fisher Information Matrix rank from a
central-finite-difference Jacobian at the literature-midpoint parameter
vector) from the 3 single-agent R values (R_ARI, R_MEM, R_BUS) already used
in the v2 likelihood, to every additional condition Tier 2 could plausibly
add a real behavioral anchor for: the three pairwise combinations
(ARI+MEM, ARI+BUS, MEM+BUS) and the triple combination.

Caveat stated explicitly in the output: the FIM rank bound (rank <=
number of real-valued outputs) is a NECESSARY condition for identifiability,
not sufficient -- two outputs can be functionally dependent (e.g. highly
correlated), in which case adding a condition does not add a full unit of
rank. This script reports the ACHIEVED rank for each candidate anchor set,
not just the naive output-count upper bound, so it is not vulnerable to that
gap.
"""
import json
import itertools

import numpy as np

from priors import PRIOR_SPECS, STRUCTURAL, FREE_PARAM_NAMES
import cct_model as cm

VEHICLE_DOSES = [0.0, 0.0, 0.0]
CONDITION_DOSES = {
    'ARI':      [1.0, 0.0, 0.0],
    'MEM':      [0.0, 1.0, 0.0],
    'BUS':      [0.0, 0.0, 1.0],
    'ARI+MEM':  [1.0, 1.0, 0.0],
    'ARI+BUS':  [1.0, 0.0, 1.0],
    'MEM+BUS':  [0.0, 1.0, 1.0],
    'ARI+MEM+BUS': [1.0, 1.0, 1.0],
}
ALL_CONDITIONS = list(CONDITION_DOSES.keys())

# Assumed per-condition target SD, matching the SD actually used for the
# existing single-agent anchors (range/4 heuristic, evidence.py), applied
# uniformly since no real pairwise/triple behavioral data exist yet. This
# is a stated assumption, not a measured value.
ASSUMED_SD = 0.05

RANK_TOL_FRACTION = 1e-6  # same threshold as fim_sloppiness.py's "numerically zero"


def literature_midpoint_vec():
    vec = []
    for name in FREE_PARAM_NAMES:
        spec = PRIOR_SPECS[name]
        lo, hi = spec[-2], spec[-1]
        vec.append((lo + hi) / 2)
    return np.array(vec)


def build_param_dict(vec):
    p = dict(STRUCTURAL)
    for name, val in zip(FREE_PARAM_NAMES, vec):
        p[name] = float(val)
    return p


def simulate_R_for_conditions(vec, conditions):
    p = build_param_dict(vec)
    try:
        E_ctrl = cm.final_E(VEHICLE_DOSES, p)
    except RuntimeError:
        return np.full(len(conditions), -999.0)
    if E_ctrl <= 1e-6:
        return np.full(len(conditions), -999.0)
    out = []
    for c in conditions:
        try:
            E_c = cm.final_E(CONDITION_DOSES[c], p)
        except RuntimeError:
            out.append(-999.0)
            continue
        out.append((E_ctrl - E_c) / E_ctrl)
    return np.array(out)


def jacobian(vec, conditions, rel_step=1e-3):
    n = len(vec)
    m = len(conditions)
    J = np.zeros((m, n))
    for j in range(n):
        h = max(abs(vec[j]) * rel_step, 1e-5)
        v_plus, v_minus = vec.copy(), vec.copy()
        v_plus[j] += h
        v_minus[j] -= h
        R_plus = simulate_R_for_conditions(v_plus, conditions)
        R_minus = simulate_R_for_conditions(v_minus, conditions)
        J[:, j] = (R_plus - R_minus) / (2 * h)
    return J


def fim_rank(vec, conditions):
    J = jacobian(vec, conditions)
    J_weighted = J / ASSUMED_SD
    FIM = J_weighted.T @ J_weighted
    eigvals = np.linalg.eigvalsh(FIM)
    eigvals = np.clip(eigvals, 0, None)
    eigvals_sorted = np.sort(eigvals)[::-1]
    threshold = RANK_TOL_FRACTION * eigvals_sorted[0] if eigvals_sorted[0] > 0 else 0
    achieved_rank = int(np.sum(eigvals_sorted > threshold))
    return achieved_rank, eigvals_sorted


def main():
    vec = literature_midpoint_vec()
    n_params = len(vec)

    candidate_sets = [
        ('current (3 single-agent anchors)', ['ARI', 'MEM', 'BUS']),
        ('+ 1 pairwise (ARI+MEM)', ['ARI', 'MEM', 'BUS', 'ARI+MEM']),
        ('+ all 3 pairwise', ['ARI', 'MEM', 'BUS', 'ARI+MEM', 'ARI+BUS', 'MEM+BUS']),
        ('+ triple combo only (no pairwise)', ['ARI', 'MEM', 'BUS', 'ARI+MEM+BUS']),
        ('+ all 3 pairwise + triple (all 7 conditions)', ALL_CONDITIONS),
    ]

    results = []
    for label, conditions in candidate_sets:
        rank, eigvals = fim_rank(vec, conditions)
        results.append({
            'anchor_set': label,
            'conditions': conditions,
            'n_conditions': len(conditions),
            'achieved_fim_rank': rank,
            'naive_upper_bound_rank': min(len(conditions), n_params),
            'fully_identifies_all_14': rank >= n_params,
            'eigenvalues_desc': [float(v) for v in eigvals],
        })
        print(f"{label}: {len(conditions)} conditions -> achieved rank {rank}/{n_params} "
              f"(naive upper bound {min(len(conditions), n_params)})")

    with open('results/fim_anchor_requirements.json', 'w') as f:
        json.dump({
            'method': 'FIM rank at literature-midpoint parameter vector, '
                      'central finite differences, assumed target SD=0.05 '
                      'for all hypothetical pairwise/triple anchors '
                      '(no real data yet -- placeholder matching existing '
                      'single-agent anchor SD convention)',
            'n_free_params': n_params,
            'candidate_anchor_sets': results,
        }, f, indent=2)
    print("\nSaved: results/fim_anchor_requirements.json")


if __name__ == '__main__':
    main()
