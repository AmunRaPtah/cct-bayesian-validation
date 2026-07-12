# CCT Model: Bayesian In-Silico Validation (No Wet-Lab Substitute)

**Date:** 2026-07-12
**Author:** pipeline built by Claude for Eniola Olutogun, applied to the CCT model trilogy
(Olutogun E, 2026 — Papers 1-3, patent-pending; public code at
[AmunRaPtah/cct-simulation-](https://github.com/AmunRaPtah/cct-simulation-))

## Why this exists

The CCT (Conjunctive Consolidation Threshold) model predicts that simultaneous
sub-threshold suppression of three neural axes (dopaminergic RPE, NMDAR-LTP
readiness, affective contrast) — using aripiprazole (ARI), memantine (MEM), and
buspirone (BUS) as proof-of-concept agents — blocks drug-memory consolidation
*more* than the three drugs' independent effects would predict
("super-additivity"). Paper 3 of the trilogy proposes Bayesian parameter
integration as the human-translation validation strategy. No wet-lab
experiment is available here, so this pipeline substitutes: (1) real published
pharmacology as evidence, (2) Bayesian calibration of the model's free
parameters against that evidence, and (3) posterior-predictive simulation of
the model's central claim, with full uncertainty propagation instead of a
single point estimate.

**This is not equivalent to a wet-lab replication.** It answers a narrower
question: *given everything real pharmacology tells us and everything it
doesn't, how much does the CCT architecture's qualitative prediction survive?*

## Method summary

1. **Architecture** (`cct_model.py`): ported unchanged from the public
   reference implementation (MIT-licensed, structural code only — no
   proprietary calibrated coefficients).
2. **Evidence** (`evidence.py`): six categories of real, cited pharmacology
   were searched for and compiled (see below) — dose-occupancy curves, LTP
   electrophysiology, behavioral consolidation-blockade studies, and rodent
   PK. Every number traces to a specific paper; gaps are stated explicitly,
   not papered over.
3. **Priors** (`priors.py`): 14 free parameters, each given either an
   *elicited* prior (literature Table-1 range reshaped using real external
   data) or an *uninformed* prior (flat over the Table-1 range, where no
   usable external anchor exists).
4. **Likelihood-based calibration** (`bayesian_calibration.py`): of the 14
   parameters, 11 are analytically identified by the one real datum whose
   units match the model's actual output — fractional reduction in cumulative
   encoding probability for memantine given alone during the consolidation
   window (Cox/Meyers et al. 2012, PMC3917839: 6.5 mg/kg memantine completely
   blocked cocaine-CPP consolidation). The other 3 (phi1, phi3, delta_BUS) are
   provably irrelevant to this likelihood (their only terms are multiplied by
   ARI=0 or BUS=0 in the vehicle/MEM-alone conditions) and are excluded from
   the MCMC, kept at prior. Sampling used PyMC's `DEMetropolisZ` (gradient-free,
   since the ODE forward solve isn't differentiable through PyTensor),
   4 chains x 1200 tune x 1200 draws.
5. **Posterior-predictive validation** (`posterior_predictive.py`): 800
   parameter draws each from the prior and the posterior were propagated
   through the full 2x2x2 factorial ODE simulation to get a *distribution*
   of the super-additivity index, not a point estimate.

## What the literature search actually found

Full detail in `evidence.py`. Headline points:

- **Strongest anchor:** memantine 6.5-10 mg/kg completely blocked cocaine-CPP
  consolidation/reconsolidation (Cox/Meyers et al. 2012) — falls inside the
  model's proposed 3-10 mg/kg range, and is the one behavioral endpoint used
  for likelihood calibration here.
- **A real, unresolved tension:** in freely-moving rats, memantine 5-10 mg/kg
  *enhanced* (not suppressed) hippocampal CA1 LTP (PMID 26119223) — directly
  conflicting with a naive "memantine suppresses Axis-2 amplitude" reading at
  the electrophysiological level. This calibration used the behavioral
  endpoint (matches the model's actual output variable) and treats the LTP
  finding as an open discrepancy the model does not currently explain — a
  concrete, falsifiable question for future wet-lab work.
- **D1/D5 gate:** a real, fairly dramatic threshold-crossing effect exists
  (PMC3633881: subthreshold stimulation goes from no LTP to robust LTP under
  D1/D5 priming) — used only to elicit an informed prior on gamma_g (the paper's
  a.u. units don't map cleanly onto the electrophysiology paper's EPSP-magnitude
  units, so it was not run through a formal likelihood).
- **Aripiprazole:** strong real D2 occupancy data (Mamo et al. 2007, Yokoi et
  al. 2002) supports an elicited high prior for phi1, but no rodent
  RPE-amplitude measurement exists to calibrate it quantitatively.
- **Buspirone: essentially unconstrained.** Rabiner et al. 2000 found ~0-5%
  5-HT1A occupancy at a standard clinical dose despite clear behavioral
  efficacy (low-occupancy/high-efficacy partial-agonist pharmacology) — no
  usable dose-response curve, and no buspirone-specific consolidation-blockade
  study was found (only 8-OH-DPAT, a different drug). phi3 and delta_BUS
  remain literature-range priors, not Bayes-updated.
- **PK-implied decay rates:** rodent half-lives give order-of-magnitude
  cross-checks on lambda1/2/3. Aripiprazole's is a genuine mismatch (real
  clearance faster than the manuscript's literature range implies) — flagged,
  not resolved.

## Results

| | Super-additivity index (pp) | 94% HDI | P(index > 0) |
|---|---|---|---|
| **Prior-predictive** (literature-range uncertainty only) | 22.0 | [9.8, 34.6] | 1.0000 (n=800) |
| **Posterior-predictive** (conditioned on real MEM behavioral data) | 13.2 | [6.1, 21.1] | 1.0000 (n=800) |

See `results/super_additivity_distribution.png`.

**Interpretation:** the model's central qualitative claim — that the triple
combination beats the Bliss-independence expectation from the three
single-agent effects — holds across the entire sampled parameter space, both
before and after folding in the one real quantitative behavioral anchor
available. Conditioning on real data did two things: it **shrank the
predicted effect size roughly by a third** (22.0pp -> 13.2pp) and **tightened
the uncertainty band** (sd 0.070 -> 0.040). It did not overturn the direction
of the prediction. This is the most defensible "positive" result available
without new experimental data: the architecture's super-additivity claim is
robust to realistic parameter uncertainty grounded in what real pharmacology
supports, not just to the model's own literature-range assumptions.

## What this does *not* establish

- It does not validate the *magnitude* reported in the manuscript (E_final =
  0.855, super-additivity = 12.8 pp) — those used privately-held calibrated
  coefficients not available here. (Note the posterior-predictive mean of
  13.2 pp is strikingly close to the manuscript's reported 12.8 pp, but this
  should be read as a plausibility check, not confirmation — the calibration
  target and the manuscript's fitting procedure are different.)
- It does not resolve the memantine LTP-enhancement vs. CPP-consolidation-
  blockade conflict.
- It does not touch buspirone's Axis-3 parameters with real data at all —
  that entire arm of the model remains literature-range-only.
- MCMC convergence is borderline (max r_hat = 1.05, min bulk ESS = 120 across
  4800 post-warmup draws) — acceptable for a directional conclusion, not for
  precise point estimates of individual parameters. Longer runs or a
  differentiable surrogate (e.g. a neural-network emulator of the ODE, enabling
  NUTS) would tighten this if pursued further.
- The mg/kg -> normalized-EC50 dose mapping used to place the real memantine
  evidence into the model's units is an explicit assumption (documented in
  `bayesian_calibration.py`), not a measured conversion, because the true
  mapping is held privately pending the patent filing.

## Reproducing

```
/root/projects/pardalos/.venv/bin/python bayesian_calibration.py   # ~7-8 min
/root/projects/pardalos/.venv/bin/python posterior_predictive.py   # ~5 min
```

Outputs land in `results/`: `cct_posterior.nc`, `cct_prior.nc` (full traces),
`posterior_summary.csv`, `super_additivity_summary.csv`,
`super_additivity_distribution.png`.
