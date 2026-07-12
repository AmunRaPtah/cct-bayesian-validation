"""
Prior distributions for the 14 free CCT model parameters.

Two tiers, per evidence.py:
  - INFORMED: literature Table-1 range shifted/widened using real external
    data (PK half-lives, PET occupancy, gate threshold-crossing magnitude).
    These are elicited priors -- the evidence moves the distribution, but is
    not run through a matched-units likelihood (see evidence.py for why:
    dimensional/paradigm mismatches between the real data and the model's a.u.).
  - UNINFORMED: literature Table-1 range only (Uniform), because the
    literature search found no usable external quantitative anchor.

phi2 and delta_MEM start uninformed here (flat over their Table-1 range) and
get properly Bayes-updated in bayesian_calibration.py against
evidence.MEM_CONSOLIDATION_EVIDENCE, the one place where model output units
(fractional encoding-probability reduction) match a real behavioral endpoint.

Each entry: name -> ('uniform', low, high) or ('truncnorm', mean, sd, low, high)
"""

PRIOR_SPECS = {
    # --- decay rates (h^-1) -- PK-informed, with documented conflicts ---
    'lambda1': ('truncnorm', 0.50, 0.10, 0.35, 0.70),   # widened toward faster ARI PK decay (~0.69 h^-1)
    'lambda2': ('truncnorm', 0.30, 0.06, 0.20, 0.40),   # near overlap of Table1 [.25,.40] & PK-implied [.18,.37]
    'lambda3': ('truncnorm', 0.55, 0.08, 0.40, 0.80),   # shifted toward faster BUS brain PK decay

    # --- drug suppression efficacies (a.u.) ---
    'phi1': ('truncnorm', 0.63, 0.06, 0.50, 0.70),      # elicited high (near-saturating D2 occupancy)
    'phi2': ('uniform', 0.55, 0.75),                     # uninformed prior; Bayes-updated via MEM likelihood
    'phi3': ('uniform', 0.35, 0.50),                     # uninformed -- no usable BUS data

    # --- threshold-shift parameters (a.u.) ---
    'delta_MEM': ('uniform', 0.30, 0.60),                # uninformed prior; Bayes-updated via MEM likelihood
    'delta_BUS': ('uniform', 0.35, 0.75),                # uninformed -- no usable BUS data

    # --- D1/D5 permissive gate ---
    'theta_g': ('uniform', 0.35, 0.45),                  # uninformed -- no comparable-units data
    'gamma_g': ('truncnorm', 0.32, 0.025, 0.25, 0.35),   # elicited high (large empirical threshold-crossing effect)

    # --- intrinsic encoding thresholds (a.u.) -- no external data sought ---
    'theta1': ('uniform', 0.40, 0.60),
    'theta2_base': ('uniform', 0.40, 0.60),
    'theta3': ('uniform', 0.25, 0.40),

    # --- encoding rate (h^-1) -- free calibration parameter, no external data ---
    'r_E': ('uniform', 1.5, 3.0),
}

# Structural constants held fixed (paper explicitly calls k_g/k1/k2/k3
# "structural, not proprietary"; mu1/mu2/mu3 are normalization gains with no
# literature range given -- fixed at the manuscript's own default of 0.5).
STRUCTURAL = {
    'k_g': 5.0, 'k1': 4.0, 'k2': 4.0, 'k3': 4.0,
    'mu1': 0.5, 'mu2': 0.5, 'mu3': 0.5,
}

FREE_PARAM_NAMES = list(PRIOR_SPECS.keys())

# phi1, phi3, delta_BUS are analytically unidentified by the MEM-consolidation
# likelihood used in bayesian_calibration.py: the vehicle and MEM-alone
# conditions both set ARI=0 and BUS=0, which exactly zeroes out every term
# these three parameters appear in (phi1 only multiplies ARI, phi3 only
# multiplies BUS, delta_BUS only multiplies BUS). Including them in the MCMC
# wastes sampler dimensions with zero likelihood gradient and hurts mixing.
# They are sampled from prior only and merged in at the posterior-predictive
# stage (see posterior_predictive.py), which does exercise ARI/BUS doses.
UNIDENTIFIED_BY_MEM_LIKELIHOOD = ['phi1', 'phi3', 'delta_BUS']
IDENTIFIED_PARAM_NAMES = [n for n in FREE_PARAM_NAMES if n not in UNIDENTIFIED_BY_MEM_LIKELIHOOD]


def scipy_frozen_dists():
    """Return {name: scipy.stats frozen distribution} for prior-predictive use."""
    from scipy import stats
    dists = {}
    for name, spec in PRIOR_SPECS.items():
        kind = spec[0]
        if kind == 'uniform':
            _, lo, hi = spec
            dists[name] = stats.uniform(loc=lo, scale=hi - lo)
        elif kind == 'truncnorm':
            _, mean, sd, lo, hi = spec
            a, b = (lo - mean) / sd, (hi - mean) / sd
            dists[name] = stats.truncnorm(a, b, loc=mean, scale=sd)
        else:
            raise ValueError(kind)
    return dists
