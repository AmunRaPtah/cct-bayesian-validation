# cct-bayesian-validation

Bayesian in-silico validation pipeline for the CCT (Conjunctive Consolidation
Threshold) model, standing in for wet-lab replication.

See `REPORT.md` for methodology, evidence sources, results, and limitations.

## Files

- `cct_model.py` — CCT ODE forward simulator (architecture ported from the
  public reference implementation, MIT-licensed).
- `evidence.py` — real, cited pharmacology data used to ground priors/likelihood.
- `priors.py` — prior distributions for the model's 14 free parameters.
- `bayesian_calibration.py` — PyMC calibration against a real behavioral
  consolidation-blockade endpoint (memantine/cocaine-CPP).
- `posterior_predictive.py` — propagates posterior uncertainty through the
  full 2x2x2 factorial simulation to validate the super-additivity claim.
- `fim_sloppiness.py` — structural identifiability / sloppy-model analysis:
  Fisher Information Matrix eigenspectrum for the v2 (3-drug) likelihood.
- `map_diagnostic.py` — independent trust-region MAP-finding diagnostic
  (scipy least_squares/trf, 25 restarts) against the same v2 likelihood.
- `sobol_sensitivity.py` — global Sobol sensitivity analysis (SALib) over all
  14 parameters' full literature-range bounds, independent of MCMC convergence.
- `sobol_second_order.py` — second-order Sobol index for the gate parameter
  pair (gamma_g x theta_g) specifically, testing Section 3.3's analytic
  cross-partial claim against a variance-based interaction measure.
- `phi2_zero_robustness.py` — robustness check with MEM's amplitude-suppression
  channel (phi2) pinned at 0, addressing the Ma et al. 2015 LTP-enhancement
  tension.
- `results/` — traces, summaries, figures, JSON diagnostic outputs.

## Environment

Uses the Pardalos scientific Python stack:
`/root/projects/pardalos/.venv/bin/python` (pymc, arviz, scipy, matplotlib,
SALib).

## Run order

```
python bayesian_calibration.py
python posterior_predictive.py
python fim_sloppiness.py
python map_diagnostic.py
python sobol_sensitivity.py 1024 4
python sobol_second_order.py 512 4
python phi2_zero_robustness.py
```
