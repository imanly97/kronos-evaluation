# Hourly-RV pilot

The first check on whether Kronos-small forecasts realised volatility on hourly
equity bars, given that it has *no* directional skill on daily bars. Motivated
the full 30-name study in [`notebooks/02_vol_forecast_eval.ipynb`](../../notebooks/02_vol_forecast_eval.ipynb) —
trust that notebook where the two disagree.

- `probe_hourly_rv.py` — the probe script (self-contained, uses only `src/kronos.py`)
- `trajectories.py` — re-runs 6 forecasts keeping the full sample paths (for the
  fan-chart plots; the probe only saved summary RV)
- `data/probe_results.pkl` — 960 forecasts: 8 names × 120 origins (aligned to
  688 on a common origin grid in the notebook), Oct 2024 → Aug 2026, fully
  out-of-sample (Kronos pretraining ends ~June 2024). Columns:
  `ticker, origin, realized, kronos, ewma, har, naive`
- `data/hourly_cache/*.pkl` — the raw hourly OHLCV pulled for each name
- `data/trajectories.pkl` — 6 forecasts with full paths kept

Analysis: [`notebooks/probe_findings.ipynb`](../../notebooks/probe_findings.ipynb).

## What it found

- Kronos adds statistically-robust incremental volatility information over EWMA
  (+0.05 log-RV R² on this sample) — enough to justify the full study.
- Modest within-name skill (Spearman ~0.31 vs EWMA 0.24), concentrated on
  eventful / higher-vol names.
- Miscalibrated on level (biased low); a cheap expanding-window affine
  recalibration fixes most of it.
- A vol-targeting / Sharpe check was inconclusive on this small a sample (see
  the notebook — an earlier version of this check had an origin-alignment bug
  that produced a spurious significant result; corrected).

**The full study (30 names, real HAR-RV/GARCH baselines, Diebold–Mariano tests,
Model Confidence Set) found Kronos loses to HAR-RV overall** — see the top-level
[`README.md`](../../README.md) for the actual conclusion. This pilot's value was
in motivating that study and in the two sample-path cases: Kronos correctly
reading a stale volatility spike as over (AMD, 2025-04-11) vs. completely
missing a scheduled-earnings gap it has no way to see (TSLA, 2025-07-23).
