# Evaluating Kronos for equity price forecasting

Does [Kronos](https://github.com/shiyu-coder/Kronos) — an open-source foundation
model for financial candlesticks — have genuine, baseline-beating forecasting
skill on US equities? This repo is a rigorous, reproducible answer: a **map** of
where it works and where it fails, across horizon, frequency, target (realised
volatility / distribution calibration / return direction) and regime, against the
right classical baseline in each cell, with the statistics to defend it.

Kronos is a **frozen input** — no training, no fine-tuning.

Full plan, methodology, and splits: [`PLAN.md`](PLAN.md).

## What we know so far (the pilot)

`research/probe_hourly_rv/` — 8 names, 960 forecasts, out of sample:

- **No directional skill** on daily bars (49.5% sign hit vs 54.6% base rate).
- **Modest realised-vol skill** on hourly bars — +0.05 incremental log-RV R² over
  EWMA (bootstrap CI excludes zero); edge concentrated on high-vol names and
  turbulent regimes.
- **Miscalibrated on level** (biased low); a cheap affine recalibration fixes it.
- **Overconfident intervals** (daily 90% band covers ~78%).
- Big misses cluster on **price gaps** — news/earnings the price-only model can't
  see. Deferred to a follow-up study.

See [`notebooks/probe_findings.ipynb`](notebooks/probe_findings.ipynb).

## Splits

Kronos-small pretraining ends **~June 2024** (arXiv:2508.02739). The splits
control *our* methodology overfitting, not the model's:

| window | dates | use |
|---|---|---|
| contaminated | ≤ 2024-06-30 | daily only; measure the contamination gap |
| OOS-development | 2024-07-01 → 2025-12-31 | all methodology choices |
| OOS-lockbox | 2026-01-01 → 2026-08-31 | run once, report |

## Quick start

```bash
scripts/setup.sh                       # venv + deps + vendored Kronos + weights
.venv/bin/python -m pytest -q          # no-lookahead + metric sanity tests
```

## Layout

```
PLAN.md                     the study plan (questions, methodology, splits)
docs/PLAN_v1_agentic_desk.md the archived prior direction
src/
  kronos.py                 the dispersion-preserving sampler (the only Kronos code we own)
  data.py                   no-lookahead price loader (daily + hourly)
  config.py                 paths, universe, splits, model params
  targets.py                RV / direction / quantile target construction     [W1]
  baselines.py              EWMA, HAR-RV, GARCH, RW — all walk-forward         [W1]
  forecast.py               run Kronos over a grid of origins -> forecast store [W1]
  metrics.py                QLIKE, pinball, CRPS, PIT, MZ, Diebold–Mariano, MCS [W1]
  recalibrate.py            the expanding-window affine layer                  [W2]
  evaluate.py               the harness                                        [W2]
research/probe_hourly_rv/   the pilot study
notebooks/                  01_data … 05_mechanism
report/findings.md          the writeup
```

## The one non-obvious thing

`KronosPredictor.predict(sample_count=N)` runs N sampled paths and then
**averages them** before returning — a smoothed mean line, zero dispersion.
`src/kronos.py::_auto_regressive_paths` is the identical batched inference
returning the array *before* the mean. That is the entire predictive
distribution, and the distribution is what we're evaluating.
