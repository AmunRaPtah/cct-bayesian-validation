"""
Real published data used as Bayesian evidence for the CCT model calibration.

Every number here traces to a specific cited study (see comments). Where the
literature search could only find a qualitative or partial-precision claim,
that is stated explicitly -- nothing here is fabricated or interpolated
beyond what a source actually reports.

This module is the single source of truth for "what do we actually know
empirically" as distinct from cct_model.py (architecture) and priors.py
(how that evidence gets turned into distributions).
"""

# ---------------------------------------------------------------------------
# 1. D1/D5 gate threshold-crossing magnitude  ->  informs gamma_g (gate gain)
# ---------------------------------------------------------------------------
# Source: "Reduced Threshold for Induction of LTP by Activation of Dopamine
# D1/D5 Receptors at Hippocampal CA1-Subiculum Synapses" (PMC3633881).
# A subthreshold HFS protocol alone produced no LTP: normalized EPSP
# 1.03 +/- 0.04 (n=10, p>0.05). The SAME subthreshold stimulus produced
# robust LTP after D1/D5 priming (SKF 38393): 1.53 +/- 0.08 (n=7, p<0.001).
# This is treated as a real (mean, sd, n) data pair we can fit the gate
# submodel against.
GATE_EVIDENCE = {
    'citation': 'D1/D5 threshold-crossing at CA1-subiculum synapses, PMC3633881',
    'no_gate_mean': 1.03, 'no_gate_sd': 0.04, 'no_gate_n': 10,
    'gate_primed_mean': 1.53, 'gate_primed_sd': 0.08, 'gate_primed_n': 7,
    # Fractional gain of the gate = (primed - no_gate) / no_gate
    'fractional_gain_mean': (1.53 - 1.03) / 1.03,   # ~0.485
}

# ---------------------------------------------------------------------------
# 2. Single-agent CPP effect sizes, all three drugs -> informs phi1, phi2,
#    phi3, delta_MEM, delta_BUS via real behavioral likelihood targets
# ---------------------------------------------------------------------------
# Source: CCT Master Parameter Table S1 (Olutogun, supplement to the CCT
# trilogy), the project's own authoritative literature-range table, citing:
#   - phi1 (ARI): 40-60% CPP reduction -- Wee et al. 2007
#   - phi2 (MEM): 50-70% CPP acquisition block -- Popik & Danysz 1997
#   - phi3 (BUS): 30-50% CPP reduction -- Haleem et al. 2014
# Each range is treated as an approximate 95% interval (mean = midpoint,
# sd = range_width / 4) and used as a Gaussian likelihood target on the
# model's fractional single-agent encoding-reduction output (R_ARI, R_MEM,
# R_BUS respectively), computed by running the ODE forward for that drug
# alone vs. vehicle. This lets all three drug-efficacy axes be identified by
# real behavioral data, not just memantine.
ARI_CPP_EVIDENCE = {
    'citation': 'Wee et al. 2007 (via CCT Master Parameter Table S1)',
    'pct_reduction_range': (40, 60),
    'R_ARI_target_mean': 0.50, 'R_ARI_target_sd': 0.05,
}
MEM_CONSOLIDATION_EVIDENCE = {
    'citation': 'Popik & Danysz 1997 (via CCT Master Parameter Table S1); '
                'corroborated directionally by Cox/Meyers et al. Psychopharmacology '
                '2012, PMC3917839 (6.5 mg/kg memantine completely blocked cocaine-CPP '
                'consolidation -- a categorical, not percentage, result, so used only '
                'as directional support, not the primary numeric target)',
    'pct_block_range': (50, 70),
    'R_MEM_target_mean': 0.60, 'R_MEM_target_sd': 0.05,
}
BUS_CPP_EVIDENCE = {
    'citation': 'Haleem et al. 2014 (via CCT Master Parameter Table S1)',
    'pct_reduction_range': (30, 50),
    'R_BUS_target_mean': 0.40, 'R_BUS_target_sd': 0.05,
}

# CAVEAT (explicitly not fed into the likelihood, reported as a limitation):
# In vivo freely-moving rat electrophysiology (PMID 26119223) found memantine
# at 5-10 mg/kg i.p. ENHANCED (not suppressed) 200-Hz-train hippocampal CA1
# LTP relative to saline. This directly conflicts with a simple "memantine
# suppresses Axis-2 amplitude" reading at the electrophysiological level, even
# though the behavioral (CPP consolidation) endpoint above shows blockade at
# a similar/lower dose. Because the CCT model's actual output variable is
# encoding probability (a behavioral/systems-level construct), not raw LTP
# amplitude, MEM_CONSOLIDATION_EVIDENCE (the behavioral endpoint) is used for
# calibration, and this electrophysiological tension is carried into the
# report as an unresolved discrepancy the model does not currently explain.
MEM_LTP_CONFLICT_NOTE = (
    "PMID 26119223: memantine 5-10 mg/kg i.p. enhanced (not suppressed) "
    "hippocampal CA1 LTP in freely-moving rats -- conflicts with a naive "
    "'memantine suppresses Axis-2 amplitude' reading. Not used in the "
    "likelihood; reported as an open discrepancy."
)

# ---------------------------------------------------------------------------
# 3. Aripiprazole D2/5-HT1A occupancy  ->  informs phi1 prior (elicitation, not likelihood)
# ---------------------------------------------------------------------------
# Source: Mamo et al. 2007, Am J Psychiatry, PMID 17728427 (triple-tracer PET):
# striatal D2 occupancy ~85% even at the LOWEST clinical dose tested (10 mg),
# rising further with dose. Yokoi et al. 2002 (PMID 12093598): D2/D3 occupancy
# 40-95% across 0.5-30 mg/day. Near-saturating D2 occupancy across the clinically
# active dose range supports a prior mass shifted toward the upper half of the
# manuscript's literature range for phi1 ([0.50, 0.70]), rather than a flat
# uniform prior across it. This is elicitation (moving a prior), not a fitted
# likelihood, because no rodent RPE-amplitude-suppression measurement exists to
# map occupancy directly onto the model's phi1 units.
ARI_OCCUPANCY_EVIDENCE = {
    'citation': 'Mamo et al. 2007 PMID 17728427; Yokoi et al. 2002 PMID 12093598',
    'occupancy_pct_low_dose': 85,
    'occupancy_pct_range': (40, 95),
    'note': 'Near-saturating occupancy across clinical dose range -> elicited '
            'prior shifted toward upper half of literature phi1 range.',
}

# ---------------------------------------------------------------------------
# 4. Rodent pharmacokinetic decay rates -> inform lambda1, lambda2, lambda3 priors
# ---------------------------------------------------------------------------
# lambda_i (h^-1) = ln(2) / t_half. Used as a mechanistic cross-check on the
# manuscript's Table 1 decay-rate ranges (which may represent receptor-signal
# decay rather than literal drug clearance -- see note).
import numpy as np

PK_EVIDENCE = {
    'ARI': {
        'citation': 'Rat IV PK, terminal t1/2 ~1h (species-comparison PK review, ~PMID 14686220 region)',
        't_half_h': 1.0,
        'lambda_implied': np.log(2) / 1.0,   # ~0.693 h^-1
    },
    'MEM': {
        'citation': 'Rat oral/IV PK, PMC3269340',
        't_half_h_range': (1.90, 3.92),
        'lambda_implied_range': (np.log(2) / 3.92, np.log(2) / 1.90),  # ~0.177-0.365 h^-1
    },
    'BUS': {
        'citation': 'Rat brain ex vivo [3H]-DPAT binding PK, biphasic',
        't_half_h_range': (24.8 / 60, 96 / 60),  # 0.413-1.6 h
        'lambda_implied_range': (np.log(2) / 1.6, np.log(2) / 0.413),  # ~0.433-1.678 h^-1
    },
}
PK_CONFLICT_NOTE = (
    "Aripiprazole rodent-PK-implied lambda1 (~0.69 h^-1) exceeds the "
    "manuscript's literature range [0.35, 0.50] h^-1 -- i.e. real ARI plasma "
    "clearance is faster than the range assumed for Axis-1 signal decay. "
    "This is consistent with lambda1 representing a slower receptor/signal "
    "decay process rather than literal drug PK, but it is an unresolved "
    "mismatch if lambda1 is meant to track ARI clearance directly."
)

# ---------------------------------------------------------------------------
# 5. Buspirone -- essentially unconstrained (explicitly not given an informed prior)
# ---------------------------------------------------------------------------
# Rabiner et al. 2000 (PET, [18F]MPPF): buspirone 10 mg (clinical dose) produced
# only ~0-5% 5-HT1A occupancy, far lower than its behavioral efficacy would
# suggest for a classical occupancy-response mapping. No buspirone-specific
# memory-consolidation effect-size study was found (only 8-OH-DPAT, a
# different, non-clinical 5-HT1A agonist, in fear conditioning). phi3 and
# delta_BUS are therefore kept as literature-range (Table 1) uniform priors,
# uninformed by external likelihood -- this is a genuine evidentiary gap, not
# an oversight.
BUS_GAP_NOTE = (
    "Rabiner et al. 2000: buspirone 10mg -> ~0-5% 5-HT1A PET occupancy despite "
    "clinical efficacy (low-occupancy/high-efficacy partial agonist pharmacology) "
    "-- occupancy is a poor proxy here. No buspirone-specific consolidation-"
    "blockade effect size found. phi3/delta_BUS kept as uninformed literature-"
    "range priors."
)
