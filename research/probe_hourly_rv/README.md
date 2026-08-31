# Hourly-RV feasibility probe

Go/no-go check before the v3 pivot: **does Kronos-small forecast realised
volatility on hourly equity bars** (the timescale and moment it was built for),
given that it has *no* directional skill on daily bars?

- `probe_hourly_rv.py` — the probe script (self-contained, doesn't touch `src/`)
- `trajectories.py` — re-runs 6 forecasts keeping the full sample paths (for the
  fan-chart plots; the probe only saved summary RV)
- `data/probe_results.pkl` — 960 forecasts: 8 names × 120 origins, Oct 2024 → Aug
  2026, fully out-of-sample (Kronos pretraining ends ~June 2024). Columns:
  `ticker, origin, realized, kronos, ewma, har, naive`
- `data/hourly_cache/*.pkl` — the raw hourly OHLCV pulled for each name
- `data/trajectories.pkl` — 6 forecasts with full paths kept

Analysis: [`notebooks/probe_findings.ipynb`](../../notebooks/probe_findings.ipynb).

**Verdict: GO.**
- Kronos adds statistically-robust incremental volatility information over EWMA
  (+0.053 log-RV R², 95% CI [+0.028, +0.084]).
- Modest within-name (Spearman ~0.29 vs EWMA 0.22), concentrated on eventful /
  higher-vol names and in the elevated/turbulent regime.
- Miscalibrated on level (MZ slope 0.82, biased low); a cheap expanding-window
  affine recalibration fixes QLIKE (1.9 → 1.1, below EWMA), ranking unchanged.
- Modestly **improves vol targeting**: raw-Kronos inverse-vol sizing tightens
  delivered-risk CV ~2.5% vs constant weighting and beats EWMA, which *worsens*
  control here.

Small effects, but real, robust, and Kronos beats the standard practitioner
baseline (EWMA). It ranks which names will have a noisy session — which is what a
triage desk needs.
