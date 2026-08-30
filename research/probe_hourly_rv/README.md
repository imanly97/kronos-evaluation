# Hourly-RV feasibility probe

Go/no-go check before the v3 pivot: **does Kronos-small forecast realised
volatility on hourly equity bars** (the timescale and moment it was built for),
given that it has *no* directional skill on daily bars?

- `probe_hourly_rv.py` — the probe script (self-contained, doesn't touch `src/`)
- `data/probe_results.pkl` — 960 forecasts: 8 names × 120 origins, Oct 2024 → Aug
  2026, fully out-of-sample (Kronos pretraining ends ~June 2024). Columns:
  `ticker, origin, realized, kronos, ewma, har, naive`
- `data/hourly_cache/*.pkl` — the raw hourly OHLCV pulled for each name

Analysis: [`notebooks/probe_findings.ipynb`](../../notebooks/probe_findings.ipynb).

**Verdict: GO.** Kronos adds statistically-robust incremental volatility
information over EWMA (+0.053 log-RV R², 95% CI [+0.028, +0.084]); the edge is
modest within-name (~0.07 Spearman) and concentrates on eventful, higher-vol
names in the elevated/turbulent regime; it needs a cheap affine level
recalibration (fixes QLIKE 1.9 → 1.1). It is a vol-*expansion anticipation*
signal, not a general vol model — which is what a triage desk ranks on.
