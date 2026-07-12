"""
CCT model forward simulator (ODE system).

Adapted from the public reference implementation:
  Olutogun E (2026). CCT Model: Conjunctive Consolidation Threshold.
  https://github.com/AmunRaPtah/cct-simulation-  (MIT License)

Structural/architectural code only (sigmoid gating, ODE right-hand side,
factorial runner, super-additivity metric). No proprietary calibrated
coefficients are hard-coded here -- all parameters are passed in and are
either literature-range priors or posterior samples produced by this
validation pipeline.
"""

import numpy as np
from scipy.integrate import solve_ivp
from itertools import product as iproduct

THERAPEUTIC_DOSES = {'ARI': 0.5, 'MEM': 0.5, 'BUS': 0.5}  # normalized to EC50, placeholder


def sigmoid(x, k, theta):
    return 1.0 / (1.0 + np.exp(-k * (x - theta)))


def G(A1, p):
    """D1/D5 permissive gate: sigmoid(A1, k_g, theta_g)."""
    return sigmoid(A1, p['k_g'], p['theta_g'])


def theta2_eff(A1, p, MEM=0.0):
    gate_mod = p['gamma_g'] * G(A1, p)
    mem_thresh = p['delta_MEM'] * MEM
    return p['theta2_base'] - gate_mod + mem_thresh


def theta3_eff(p, BUS=0.0):
    return p['theta3'] + p['delta_BUS'] * BUS


def P_encoding(A1, A2, A3, p, drug_doses):
    ARI, MEM, BUS = drug_doses
    s1 = sigmoid(A1, p['k1'], p['theta1'])
    s2 = sigmoid(A2, p['k2'], theta2_eff(A1, p, MEM))
    s3 = sigmoid(A3, p['k3'], theta3_eff(p, BUS))
    return s1 * s2 * s3


def RPE(t):
    return np.exp(-0.5 * (t / 1.5) ** 2)


def STIM(t):
    return np.exp(-0.5 * (t / 2.0) ** 2)


def CONTRAST(t):
    rise = np.clip((t + 1.0) / 2.0, 0.0, 1.0)
    decay = np.exp(-0.3 * np.maximum(t - 1.0, 0.0))
    return 0.3 + 0.5 * rise * decay


def system_odes(t, y, p, drug_doses):
    A1, A2, A3, E = y
    ARI, MEM, BUS = drug_doses

    dA1 = -p['lambda1'] * A1 + p['mu1'] * RPE(t) - p['phi1'] * ARI * A1
    dA2 = -p['lambda2'] * A2 + p['mu2'] * G(A1, p) * STIM(t) - p['phi2'] * MEM * A2
    dA3 = -p['lambda3'] * A3 + p['mu3'] * CONTRAST(t) - p['phi3'] * BUS * A3

    P = P_encoding(A1, A2, A3, p, drug_doses)
    dE = p['r_E'] * P * (1.0 - E)

    return [dA1, dA2, dA3, dE]


def run_condition(drug_doses, p, t_span=(-2.0, 6.0), n_eval=120):
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    y0 = [0.01, 0.01, 0.01, 0.0]
    sol = solve_ivp(system_odes, t_span, y0, args=(p, drug_doses), t_eval=t_eval,
                     method='RK45', rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return float(sol.y[3, -1])


def final_E(drug_doses, p, t_span=(-2.0, 6.0), rtol=1e-5, atol=1e-6):
    """Lean final-state-only solve (no t_eval array) for use inside MCMC likelihoods."""
    y0 = [0.01, 0.01, 0.01, 0.0]
    sol = solve_ivp(system_odes, t_span, y0, args=(p, drug_doses),
                     method='RK45', rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return float(sol.y[3, -1])


def run_factorial(p, doses=None):
    if doses is None:
        doses = THERAPEUTIC_DOSES
    results = {}
    for ari_on, mem_on, bus_on in iproduct([False, True], repeat=3):
        d = [doses['ARI'] if ari_on else 0.0,
             doses['MEM'] if mem_on else 0.0,
             doses['BUS'] if bus_on else 0.0]
        results[(ari_on, mem_on, bus_on)] = run_condition(d, p)
    return results


def compute_super_additivity(results):
    E_ctrl = results[(False, False, False)]
    E_A1 = results[(True, False, False)]
    E_A2 = results[(False, True, False)]
    E_A3 = results[(False, False, True)]
    E_combo = results[(True, True, True)]

    R = lambda E: (E_ctrl - E) / E_ctrl if E_ctrl > 0 else 0.0
    R_A1, R_A2, R_A3 = R(E_A1), R(E_A2), R(E_A3)
    R_combo = R(E_combo)
    R_indep = 1.0 - (1.0 - R_A1) * (1.0 - R_A2) * (1.0 - R_A3)

    return {
        'E_ctrl': E_ctrl, 'E_combo': E_combo,
        'R_A1': R_A1, 'R_A2': R_A2, 'R_A3': R_A3,
        'R_combo': R_combo, 'R_indep': R_indep,
        'super_additivity_index': R_combo - R_indep,
    }


DEFAULT_STRUCTURAL = {
    # structural/steepness constants -- not treated as free parameters
    # (paper flags these as "structural, not proprietary")
    'k_g': 5.0, 'k1': 4.0, 'k2': 4.0, 'k3': 4.0,
    'mu1': 0.5, 'mu2': 0.5, 'mu3': 0.5,  # input gains normalized so *_max ~ 1.0 (paper convention)
}
