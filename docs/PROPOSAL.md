# Computational de-risking of a conjunctive combination-pharmacotherapy for drug-memory consolidation

**A dry-lab research proposal (no wet-lab work proposed)**

PI / applicant: Eniola Olutogun
Status: DRAFT 2026-10-29, references require `cite-check` before any submission
Companion code: `cct-bayesian-validation` (this repository); upstream model trilogy: CCT Papers 1–3 (Olutogun, 2026, OSF/Zenodo preprints, DOIs to confirm from OSF before submission)

---

## 0. One-paragraph summary

The Conjunctive Consolidation Threshold (CCT) model predicts that drug-associated
memories are consolidated only when three neural axes, dopaminergic reward-prediction
error, NMDAR-dependent LTP readiness, and affective contrast, are **simultaneously**
above threshold in a −2 h to +6 h peri-exposure window, and therefore that a combination
of three already-approved agents given at individually sub-threshold doses (aripiprazole,
memantine, buspirone) should block consolidation **super-additively**. A Bayesian
in-silico validation already shows this super-additivity prediction is robust to realistic
pharmacological uncertainty (posterior index 13.2 percentage points, 94% HDI [6.1, 21.1],
P(index > 0) = 1.0, conditioned on a real behavioral anchor). This project does **not** run
the animal study. It does the computational work that must come first and that currently
does not exist anywhere: (1) rebuild the calibration so its uncertainty is trustworthy
rather than directional, (2) use formal identifiability analysis to decide exactly which
bench measurements would most reduce that uncertainty, (3) deliver a pre-registered,
power-analyzed 2×2×2 factorial CPP protocol whose sample size and expected effect are
derived from the model's own posterior, and (4) turn the one documented contradiction in
the evidence base, memantine *enhances* LTP yet *blocks* CPP consolidation, into a
specific, falsifiable experiment. Deliverables are an auditable model, a ready-to-execute
protocol, and an open pre-registration, all produced on commodity compute by a single
researcher in 12 months.

---

## 1. Specific aims

**Aim 1, Trustworthy re-calibration of the CCT model.** The current three-drug
calibration is directionally sound but numerically unconverged (max r̂ = 1.35, min ESS = 10
under gradient-free DEMetropolisZ, because the ODE forward solve is not differentiable
through the sampler). Train a differentiable neural-network emulator of the CCT ODE system,
enabling NUTS/HMC sampling, and recalibrate against a systematically harvested multi-endpoint
evidence set. *Success:* r̂ < 1.01 and ESS > 400 on all identified parameters, with the
super-additivity posterior reported at publication-grade precision rather than as a
directional claim.

**Aim 2, Identifiability-driven measurement design.** Extend the existing Fisher-Information
/ Sobol sloppiness analysis to rank which of the 14 parameters are (a) well-constrained by
existing literature, (b) sloppy/irrelevant, and (c) *stiff but unmeasured*, the ones whose
absence most widens the super-additivity HDI. *Success:* a ranked list of the specific
bench measurements (which drug, which axis, which readout) that would most sharpen the
prediction per unit of experimental effort. This converts "we need data" into "we need
*these three* numbers."

**Aim 3, A pre-registered, power-analyzed validation protocol.** Produce a complete 2×2×2
factorial conditioned-place-preference (CPP) protocol (aripiprazole × memantine × buspirone,
each present/absent at the model's sub-threshold doses, 8 arms) in which the required N per
arm, the primary endpoint, the analysis plan, and the Bliss/Loewe super-additivity contrast
are all derived from the Aim-1 posterior. Register it on OSF before any bench partner is
engaged. *Success:* a protocol a rodent-addiction lab can execute without further modeling,
with a power curve showing the N needed to detect the posterior-predicted effect at 80%/90%
power.

**Aim 4, Resolve the Axis-2 discrepancy in silico and make it falsifiable.** The evidence
base contains a real, documented conflict: memantine 5–10 mg/kg *enhanced* hippocampal CA1
LTP in freely-moving rats (Ma, Mufti & Leung 2015), while the same dose range *blocked*
cocaine-CPP consolidation (Alaghband & Marshall 2012). Use the `phi2`-suppression-channel
robustness machinery already in the repo to determine whether the CCT architecture can
accommodate both facts, and specify the single electrophysiology-plus-behavior experiment
that would discriminate the candidate explanations. *Success:* either a model revision that
predicts both observations, or a precise statement of the bench test that would falsify the
model.

---

## 2. Significance

Relapse in substance-use disorder is driven by persistent, cue-triggered drug memories;
disrupting their consolidation/reconsolidation is a mechanistically motivated prevention
strategy. The CCT framing is attractive for translation because all three proposed agents
are **already approved**, so a positive combination result has an unusually short path to a
human study. But no one has tested the combination, and the field has no quantitative,
uncertainty-aware prediction of how large the combination effect should be or how many
animals it would take to detect it. This project supplies exactly that missing layer, the
computational de-risking that determines whether, and how, the expensive bench experiment
is worth running. A null result here (the super-additivity collapses under honest
calibration, or the protocol needs an infeasible N) is itself a valuable, publishable outcome
that prevents wasted animal work.

## 3. Innovation

- **A pre-registered mechanistic model that already states its own expected effect size with
  a full posterior.** Most preclinical proposals cannot quantify their anticipated effect;
  here the effect size, its uncertainty, and the implied power analysis fall directly out of
  the existing Bayesian pipeline.
- **Identifiability analysis used prospectively as experimental design**, not retrospectively
  as a caveat, deciding which bench numbers to buy before any are bought.
- **A differentiable emulator** of a non-differentiable ODE consolidation model, enabling
  gradient-based inference where the field currently uses slow, poorly-converging
  gradient-free samplers.
- **An honest discrepancy treated as the central question**, not hidden. The applicant's
  track record includes pre-registering hypotheses and killing his own when they failed
  (the related Topologix program); the same discipline governs this proposal.

## 4. Preliminary data

Already in hand and auditable in this repository:

- A working CCT ODE forward simulator (`cct_model.py`), ported from the MIT-licensed public
  reference implementation (structural code only).
- A fully cited evidence compilation (`evidence.py`): dose–occupancy curves, LTP
  electrophysiology, behavioral consolidation-blockade studies, and rodent PK, every number
  traced to a source, every gap stated.
- A completed Bayesian calibration against a real behavioral endpoint, memantine 6.5 mg/kg
  blocking cocaine-CPP consolidation (Alaghband & Marshall 2012, Psychopharmacology
  226(4):707–719, PMC3917839).
- **Headline preliminary result:** the super-additivity index is 22.0 pp under
  literature-range priors and **13.2 pp (94% HDI [6.1, 21.1], P > 0 = 1.0)** after
  conditioning on the real memantine data, the prediction survives honest uncertainty, and
  conditioning on real data shrank the effect by ~1/3 and tightened the band (sd 0.070 →
  0.040).
- Structural-identifiability and global-sensitivity tooling already built: Fisher-Information
  eigenspectrum (`fim_sloppiness.py`), trust-region MAP diagnostic with 25 restarts
  (`map_diagnostic.py`), first- and second-order Sobol indices (`sobol_sensitivity.py`,
  `sobol_second_order.py`), and the `phi2`-pinned robustness check (`phi2_zero_robustness.py`).
- A documented negative/limitation ledger (REPORT.md §"What this does not establish"),
  including the known v2 convergence failure that Aim 1 exists to fix, the untouched
  buspirone arm, and the LTP discrepancy that Aim 4 targets.

Three sole-authored 2026 preprints (CCT Papers 1–3) provide the model derivation and the
proposed Bayesian-translation strategy this proposal executes.

## 5. Approach

**Aim 1.** Sample the CCT ODE over the full 14-parameter literature range; train a
neural-network emulator (e.g. an MLP or neural-ODE surrogate) on the mapping from parameters
to the factorial endpoint vector; validate the emulator out-of-sample; then run NUTS in PyMC
against the three-drug likelihood (ARI/MEM/BUS-alone CPP reductions: Wee et al. 2007; Popik &
Danysz 1997; Haleem et al. 2014). Report posterior-predictive super-additivity with proper
convergence diagnostics. Compute budget is modest (single-GPU emulator training, hours).

**Aim 2.** On the converged posterior, compute per-parameter sensitivity of the
super-additivity HDI width; cross-check the FIM eigenspectrum against the Sobol indices;
rank candidate bench measurements by expected-information-gain per experimental unit.

**Aim 3.** Translate the model's sub-threshold doses into a dosing table; derive the primary
CPP endpoint and its variance from the posterior predictive; run a Monte-Carlo power analysis
over the 2×2×2 design (the simulation machinery already exists) to produce N-per-arm at
80%/90% power for the Bliss-independence super-additivity contrast; write the full protocol
and analysis plan; pre-register on OSF.

**Aim 4.** Use `phi2_zero_robustness.py` as the starting point: test whether an architecture
in which memantine's behavioral block operates through a channel orthogonal to its
LTP-amplitude effect can reproduce both Ma 2015 and Alaghband & Marshall 2012; if so, specify
the combined LTP + CPP experiment that discriminates this from the naive single-channel
reading; if not, state the model revision or the falsification.

## 6. Timeline (12 months, single researcher)

| Months | Work |
|---|---|
| 1–3 | Aim 1 emulator + NUTS recalibration; publication-grade posterior |
| 3–5 | Aim 2 identifiability-driven measurement ranking |
| 5–8 | Aim 3 protocol, power analysis, OSF pre-registration |
| 8–11 | Aim 4 discrepancy resolution and falsifiable bench test |
| 11–12 | Open release (code, protocol, prereg), manuscript |

## 7. Budget (indicative, dry-lab only)

- Researcher stipend/salary (1 FTE, 12 mo), the dominant line.
- Cloud GPU compute for emulator training and MCMC, low (hundreds of GPU-hours; the
  existing stack runs on commodity hardware / Modal burst).
- Open-access publication fee (1 article).
- No animals, no reagents, no wet-lab facility, no bench personnel are requested.

## 8. Risks and mitigations

- *The emulator may not fit the ODE well enough for valid inference.* Mitigation: validate
  out-of-sample before trusting the posterior; fall back to longer gradient-free runs for a
  directional-only result (still a deliverable).
- *Honest calibration may collapse the super-additivity effect.* This is an acceptable,
  publishable outcome, the proposal is framed to make a null informative.
- *No bench partner may ever execute the protocol.* The protocol + prereg + model are
  self-contained scholarly outputs and a strong basis for a later collaborative wet-lab grant.

## 9. Open-science commitment

All code, the calibrated posterior, the protocol, and the pre-registration are released
openly (OSF + GitHub), continuing the applicant's existing practice with the CCT trilogy and
the Topologix program.

## References

*(To be completed and verified with `cite-check FILE` before submission, the following are
the anchors already verified in `evidence.py` / REPORT.md; the preprint DOIs must be confirmed
from OSF.)*

- Alaghband & Marshall 2012, Psychopharmacology 226(4):707–719, PMC3917839.
- Ma, Mufti & Leung 2015, Neurobiology of Aging 36(9):2544–2554, PMID 26119223.
- Mamo et al. 2007, Am J Psychiatry, PMID 17728427.
- Yokoi et al. 2002, PMID 12093598.
- Rabiner et al. 2000 ([18F]MPPF PET, buspirone 5-HT1A occupancy).
- Wee et al. 2007; Popik & Danysz 1997; Haleem et al. 2014 (single-agent CPP effect sizes, via CCT Master Parameter Table S1).
- Reduced threshold for induction of LTP by activation of dopamine D1/D5 receptors, PMC3633881.
- Olutogun E, 2026, CCT Papers 1–3 (OSF/Zenodo preprints; DOIs to confirm).
