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
- `results/` — traces, summaries, figures.

## Environment

Uses the Pardalos scientific Python stack:
`/root/projects/pardalos/.venv/bin/python` (pymc, arviz, scipy, matplotlib).

## Run order

```
python bayesian_calibration.py
python posterior_predictive.py
```
