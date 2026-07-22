"""
Regenerate Paper 2's Figure 1 and Figure 2 from real pipeline output,
replacing the versions built around the disavowed decoy figures
(E_final=0.855, super-additivity=12.8pp).

Figure 1: P(t) and E(t) trajectories over the peri-exposure window, using the
v1 posterior MEAN of the identified parameters (phi2, delta_MEM, lambda1-3,
theta_g, gamma_g, theta1, theta2_base, theta3, r_E) plus the LITERATURE-RANGE
PRIOR MEAN of the three parameters not identified by the MEM-only likelihood
(phi1, phi3, delta_BUS) -- i.e. a single representative parameter set, for
illustration only. The paper's actual quantitative claims come from the full
posterior-predictive distributions in results/full_condition_summary.csv,
not from this single trajectory.

Figure 2: bar chart of E_final across all 8 factorial conditions, using the
REAL posterior-predictive means and 94% HDIs from
results/full_condition_summary.csv (v1), with the Bliss-independence
prediction for the triple combination shown as a dashed line.
"""

import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from priors import STRUCTURAL
import cct_model as cm

# --- representative parameter set: v1 posterior means + prior means for the
# three parameters unidentified by the MEM-only likelihood ---
POSTERIOR_MEAN_V1 = {
    'lambda1': 0.560, 'lambda2': 0.307, 'lambda3': 0.570,
    'phi2': 0.652, 'delta_MEM': 0.546,
    'theta_g': 0.413, 'gamma_g': 0.311,
    'theta1': 0.513, 'theta2_base': 0.547, 'theta3': 0.330,
    'r_E': 1.980,
}
PRIOR_MEAN_UNIDENTIFIED = {'phi1': 0.63, 'phi3': 0.425, 'delta_BUS': 0.55}

P = dict(STRUCTURAL)
P.update(POSTERIOR_MEAN_V1)
P.update(PRIOR_MEAN_UNIDENTIFIED)

THERAPEUTIC_DOSES = cm.THERAPEUTIC_DOSES

CONDITIONS_FIG1 = [
    ('Vehicle', (0.0, 0.0, 0.0), '#888888', '-'),
    ('ARI only', (THERAPEUTIC_DOSES['ARI'], 0.0, 0.0), '#d97f7f', '--'),
    ('MEM only', (0.0, THERAPEUTIC_DOSES['MEM'], 0.0), '#7fa8d9', '--'),
    ('BUS only', (0.0, 0.0, THERAPEUTIC_DOSES['BUS']), '#7fd9a8', '--'),
    ('Triple combination', (THERAPEUTIC_DOSES['ARI'], THERAPEUTIC_DOSES['MEM'], THERAPEUTIC_DOSES['BUS']), '#222222', '-'),
]


def trajectory(drug_doses, p, t_span=(-2.0, 6.0), n_eval=400):
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    y0 = [0.01, 0.01, 0.01, 0.0]
    sol = solve_ivp(cm.system_odes, t_span, y0, args=(p, drug_doses), t_eval=t_eval,
                     method='RK45', rtol=1e-7, atol=1e-9)
    A1, A2, A3, E = sol.y
    Pt = np.array([cm.P_encoding(A1[i], A2[i], A3[i], p, drug_doses) for i in range(len(t_eval))])
    return t_eval, Pt, E


def make_figure1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, doses, color, ls in CONDITIONS_FIG1:
        t, Pt, E = trajectory(doses, P)
        axes[0].plot(t, Pt, label=label, color=color, linestyle=ls, linewidth=2 if 'Triple' in label or 'Vehicle' in label else 1.5)
        axes[1].plot(t, E, label=label, color=color, linestyle=ls, linewidth=2 if 'Triple' in label or 'Vehicle' in label else 1.5)
    for ax, title, ylabel in [(axes[0], 'A. Instantaneous encoding probability P(t)', 'P(t)'),
                              (axes[1], 'B. Cumulative encoding probability E(t)', 'E(t)')]:
        ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Time relative to peak drug effect (h)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig('results/figure1_trajectories.png', dpi=180)
    plt.close()
    print('Saved: results/figure1_trajectories.png')

    # report the endpoint values actually plotted, for the caption
    _, _, E_vehicle = trajectory((0.0, 0.0, 0.0), P)
    _, _, E_combo = trajectory((THERAPEUTIC_DOSES['ARI'], THERAPEUTIC_DOSES['MEM'], THERAPEUTIC_DOSES['BUS']), P)
    print(f"  Illustrative single-draw endpoints (posterior-mean parameter set): "
          f"E_vehicle={E_vehicle[-1]:.3f}, E_combo={E_combo[-1]:.3f}, "
          f"reduction={100*(1 - E_combo[-1]/E_vehicle[-1]):.1f}%")
    print("  (Not the paper's quantitative claim -- see full_condition_summary.csv "
          "for the posterior-predictive distribution across 800 draws.)")


def make_figure2():
    import csv
    rows = {}
    with open('results/full_condition_summary.csv') as f:
        for r in csv.DictReader(f):
            rows[(r['category'], r['name'])] = r

    order = ['vehicle', 'ARI_only', 'MEM_only', 'BUS_only', 'ARI_MEM', 'ARI_BUS', 'MEM_BUS', 'triple_combo']
    labels = ['Vehicle', 'ARI', 'MEM', 'BUS', 'ARI+MEM', 'ARI+BUS', 'MEM+BUS', 'Triple\ncombination']
    means = [float(rows[('E_final', c)]['mean']) for c in order]
    lo = [float(rows[('E_final', c)]['mean']) - float(rows[('E_final', c)]['hdi94_low']) for c in order]
    hi = [float(rows[('E_final', c)]['hdi94_high']) - float(rows[('E_final', c)]['mean']) for c in order]

    E_indep_combo_asfrac = 1 - float(rows[('derived', 'R_indep_combo')]['mean'])
    E_vehicle_mean = means[0]
    E_indep_combo = E_vehicle_mean * E_indep_combo_asfrac

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#888888'] + ['#7fa8d9'] * 3 + ['#d9b37f'] * 3 + ['#d97f7f']
    ax.bar(labels, means, yerr=[lo, hi], capsize=4, color=colors)
    ax.axhline(E_indep_combo, color='black', linestyle='--', linewidth=1.2,
               label=f'Bliss independence prediction (triple combo) = {E_indep_combo:.3f}')
    ax.set_ylabel('Posterior-predictive mean E_final (94% HDI)')
    ax.set_title('CCT model: posterior-predictive E_final across 2x2x2 factorial conditions\n'
                  '(v1 calibration, MEM behavioral anchor; n=800 draws)')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('results/figure2_factorial_bars.png', dpi=180)
    plt.close()
    print('Saved: results/figure2_factorial_bars.png')


if __name__ == '__main__':
    make_figure1()
    make_figure2()
