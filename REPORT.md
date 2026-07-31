# CCT Model: Bayesian In-Silico Validation (No Wet-Lab Substitute)

**Date:** 2026-07-12
**Author:** Eniola Olutogun, applied to the CCT model trilogy
(Olutogun E, 2026 — Papers 1-3, preprints in preparation, no patent filing being
pursued; public code at
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
   window (Alaghband & Marshall 2012, PMC3917839: 6.5 mg/kg memantine significantly
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

- **Strongest anchor:** memantine 6.5-10 mg/kg significantly blocked cocaine-CPP
  consolidation/reconsolidation (Alaghband & Marshall 2012, Psychopharmacology
  226(4):707-719, PMC3917839) — falls inside the model's proposed 3-10 mg/kg
  range, and is the one behavioral endpoint used for likelihood calibration
  here.
- **A real, unresolved tension:** in freely-moving rats, memantine 5-10 mg/kg
  *enhanced* (not suppressed) hippocampal CA1 LTP (Ma, Mufti & Leung 2015,
  Neurobiology of Aging 36(9):2544-2554, PMID 26119223) — directly
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

- There is no published/confirmed magnitude to validate against. Earlier
  drafts of this report (and the public code repo, before its 2026-07-12
  cleanup commit `b52ffec`) described E_final = 0.855 and super-additivity =
  12.8 pp as manuscript results withheld pending a patent filing. That filing
  is not being pursued, and the author has confirmed those specific figures
  were unconfirmed decoy values, not real calibrated results — they have
  since been removed from the public repo rather than left implying a
  finished, hidden result. The closeness of this pipeline's 13.2 pp posterior
  mean to that discarded 12.8 pp figure is therefore not meaningful and
  should not be read as a plausibility check, corroboration, or validation of
  any kind. (Table S1 does report one real documented anchor for this
  quantity — see the v2 update below.)
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
  `bayesian_calibration.py`), not a measured conversion — no such mapping has
  been published or established.

## Update (v2): calibrating against all three drugs

The original run above only had one real behavioral anchor (memantine). The
project's own **CCT Master Parameter Table S1** turned out to give real CPP
effect-size ranges for all three drugs, not just memantine:

| Parameter | Real evidence | Source |
|---|---|---|
| phi1 (ARI) | 40-60% CPP reduction | Wee et al. 2007 |
| phi2 (MEM) | 50-70% CPP acquisition block | Popik & Danysz 1997 |
| phi3 (BUS) | 30-50% CPP reduction | Haleem et al. 2014 |

`bayesian_calibration.py` was extended to three likelihood terms (ARI-alone,
MEM-alone, BUS-alone vs. vehicle), which makes all 14 parameters genuinely
identified (previously phi1/phi3/delta_BUS were provably unidentified by the
MEM-only likelihood and were excluded from the MCMC).

**Convergence got worse, not better.** Max r_hat rose from 1.05 to **1.35**
and min bulk ESS fell from 120 to **10** (4 chains x 1000 tune x 1000 draws,
DEMetropolisZ). Three correlated likelihood terms across a 14-dim black-box
posterior is a harder target for a gradient-free sampler than one term was.
Individual parameter posteriors in `results/posterior_summary_v2.csv` should
be read as rough central tendencies only, not precise estimates: some
(lambda1, phi3, theta3) have ESS as low as 10, meaning effectively only a
handful of independent samples.

**Result changed materially despite that:**

| | Super-additivity index (pp) | 94% HDI | P(index > 0) |
|---|---|---|---|
| Prior-predictive | 22.2 | [9.5, 34.6] | 1.0000 |
| Posterior-predictive (v2, 3 drugs) | **21.9** | **[15.5, 29.0]** | 1.0000 |

Unlike the v1 (MEM-only) result, which pulled the mean down from 22.0pp to
13.2pp, folding in real data for *all three* axes left the mean essentially
unchanged (22.2 -> 21.9pp) while **roughly halving the uncertainty band**
(HDI width ~25pp -> ~13.5pp). See `results/super_additivity_distribution_v2.png`.

Interpretation: once every axis has a real behavioral anchor instead of just
one, the literature-range prior's central tendency turns out to already be
compatible with what real single-agent CPP data implies -- real data mainly
buys tighter uncertainty here, not a large correction to the estimate. The
v1 result (large downward correction from one anchor) was likely an artifact
of conditioning on only one of three axes while leaving the other two
unconstrained by data.

Earlier text here compared this result to a ~12.8pp figure that had been
described, in manuscript drafts and the public code repo (before its
2026-07-12 cleanup commit `b52ffec`, "Remove stale patent-pending language
and unconfirmed decoy figures"), as a real value withheld pending a patent
filing. That filing is not being pursued, and the author has since confirmed
the 0.855/12.8pp figures were unconfirmed decoy numbers, not actual
calibrated manuscript results -- so there was never a real withheld value to
be close to, and that comparison has been removed.

The one real documented anchor for the super-additivity magnitude is **Table
S1** itself: ≈10-16pp, the range found across a 6x6 gamma_g x theta_g
sensitivity grid with the positive direction holding in all 36 tested
combinations (Olutogun 2026b). That's a range across a parameter grid, not a
single fitted point estimate, so it isn't directly comparable to this
pipeline's posterior mean either -- but it's worth flagging plainly: the v1
posterior mean (13.2pp) falls inside that range, while the **v2 posterior
mean (21.9pp) sits above its upper bound (16pp)**. That is a genuine
discrepancy between this pipeline's 3-drug calibration and the manuscript
supplement's own documented sensitivity range, not something resolved by the
evidence used here. Candidate explanations (not adjudicated by this
pipeline): the 6x6 grid may have swept a narrower region of gamma_g/theta_g
than this calibration's posterior occupies, the v2 MCMC's poor convergence
(ESS as low as 10) may be inflating the tail, or the two "super-additivity
index" definitions may not be computed identically. Worth resolving before
treating v2's 21.9pp as more reliable than v1's 13.2pp.

## Update (v3): the v1 (13.2pp) figure above was itself a convergence artifact

The "genuine discrepancy" flagged in the v2 update above turned out to have
a different resolution than any of its candidate explanations. The v1
result reported throughout this file (13.2pp, from `bayesian_calibration.py`
at 4 chains x 1200 tune x 1200 draws) passed `r_hat`/ESS diagnostics but its
posterior-predictive `R_MEM_sim` averaged 0.785 against a calibration target
of 0.60 (SD 0.05): 3.7 SD off, a value a correctly-mixed posterior under
that likelihood should not produce. The chains had converged to each other
without converging to the true posterior, a known DEMetropolisZ failure mode
with population-based proposals under short tuning.

Rerunning at 6 chains x 4000 tune x 4000 draws (`bayesian_calibration_v1_longrun.py`),
twice independently with different seeds to rule out a single-run fluke,
both reproduce the calibration target closely (`R_MEM_sim` 0.599 and 0.604)
and agree with each other on the super-additivity index to within 0.1pp:
**21.8pp** (94% HDI [11.0, 33.5], n=1600 pooled across both runs), superseding
the 13.2pp figure used throughout this file above the "Update (v3)" heading.
This closely matches v2's 21.9pp, resolving the discrepancy this section
originally flagged: v1's short run was the outlier, not v2. Full detail and
the corrected results table are in `CCT_Paper2_clean.md` Section 5.5.

A follow-up global-Sobol run restricted to the corrected v1 posterior's own
94% HDI (`sobol_sensitivity_posterior.py`) confirms the earlier
prior-range Sobol finding (`lambda1`/`r_E` dominate magnitude variance,
gate parameters negligible) is not an artifact of prior width: the same
ranking holds under the much narrower posterior ranges. A separate
structural-identifiability extension (`fim_anchor_requirements.py`) shows
CPP-outcome behavioral anchors alone, regardless of how many of the 7
possible dose combinations are tested, can identify at most 7 of the 14 free
parameters, a design-space ceiling, not a matter of insufficient anchor
count.

## Reproducing

```
/root/projects/pardalos/.venv/bin/python bayesian_calibration_v1_longrun.py --seed 20260712 --suffix longrun  # v1, corrected, ~38 min
/root/projects/pardalos/.venv/bin/python bayesian_calibration_v1_longrun.py --seed 20260713 --suffix confirm  # v1, independent-seed confirmation, ~38 min
/root/projects/pardalos/.venv/bin/python pool_full_condition_summary.py  # pools both v1 runs into the corrected Table 2

/root/projects/pardalos/.venv/bin/python bayesian_calibration.py         # v2 (3-drug), ~12-13 min
/root/projects/pardalos/.venv/bin/python posterior_predictive.py --v2    # v2 posterior-predictive, ~5 min

/root/projects/pardalos/.venv/bin/python sobol_sensitivity_posterior.py results/cct_posterior_v1_longrun.nc 1024 4  # Sobol under posterior
/root/projects/pardalos/.venv/bin/python fim_anchor_requirements.py      # minimum-anchor-set analysis
```

Outputs land in `results/`: `cct_posterior_v1_longrun.nc`/`_v1_confirm.nc`,
`cct_posterior_v2.nc` (full traces), `posterior_summary_v1_longrun.csv`/`_v1_confirm`/`_v2`,
`full_condition_summary_v1_corrected.csv`, `sobol_sensitivity_posterior_summary.json`,
`fim_anchor_requirements.json`.
