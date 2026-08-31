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
- Does **not** improve a vol-targeting strategy's Sharpe: aggressive inverse-vol
  sizing on this per-name signal *hurts* (ΔSharpe −0.68 vs buy-and-hold, CI
  excludes zero); a shrunk/recalibrated version is a wash. The RV signal is real
  but too weak on this short, concentrated sample to lift risk-adjusted returns.
  Where it does help is **tail-awareness** on individual forecasts.

The practical value is **ranking which names will have a noisy session** — which
is what a triage desk needs — plus **not being fooled by stale vol spikes**
(good case: AMD 2025-04-11). Its hard limit is scheduled catalysts, invisible to
a price-only model (bad case: TSLA earnings 2025-07-23) — which is exactly where
the desk's news/calendar layer earns its place.
