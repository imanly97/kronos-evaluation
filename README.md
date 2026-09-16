# Kronos Evaluation

Does [Kronos](https://github.com/shiyu-coder/Kronos) — an open-source foundation
model for financial candlesticks — have genuine, baseline-beating forecasting
skill on US equities?

**No.** Kronos-small has no directional skill, and on volatility — the one
place it shows any signal — it loses to a standard HAR-RV model. It carries a
small, statistically robust slice of information that even HAR misses, but not
enough to justify treating it as a forecasting edge. That result is the
deliverable of this repo.

## The finding

- **No directional skill.** Daily-bar sign prediction: 49.5% hit rate vs a
  54.6% "always guess up" baseline. Worse than the naive rule.
- **Loses to HAR-RV on volatility.** Hourly, 1-session-ahead realised
  volatility, 30 large-caps, 11,340 out-of-sample forecasts (Jul 2024–Dec 2025,
  strictly after Kronos's ~June-2024 pretraining cutoff): HAR-RV is the *sole*
  member of the 90% Model Confidence Set on QLIKE loss. Raw Kronos is
  statistically tied with plain EWMA; a cheap recalibration layer gets it just
  past EWMA and still short of HAR.
- **Does carry orthogonal information** — adding Kronos to HAR still lifts
  log-RV R² by +0.013 (bootstrap 95% CI [0.010, 0.015]). Real, but small.
- **Badly overconfident.** Its nominal 90% forecast interval covers realised
  moves only 68% of the time.
- **The edge is sector-structured, not random.** It beats EWMA in
  flow-driven names (Tech, Financials, Consumer Discretionary) and loses in
  event-driven ones (Health, Utilities, Materials) — consistent with a
  price-only model being blind to scheduled and unscheduled news (an earnings
  gap and a weekend news shock are both visible as clean forecast failures in
  the sample-path plots).

Full detail, plots, and hypothesis tests: **[`notebooks/`](notebooks/)**.

## Notebooks (the analysis)

- **[`02_vol_forecast_eval.ipynb`](notebooks/02_vol_forecast_eval.ipynb)** —
  the primary result. 30 names, walk-forward HAR-RV / GARCH / EWMA baselines,
  Diebold–Mariano tests, Model Confidence Set, incremental-information
  regression with bootstrap CIs, calibration (exact rank-PIT / CRPS), and cuts
  by realised-vol regime and sector.
- **[`probe_findings.ipynb`](notebooks/probe_findings.ipynb)** — the earlier
  8-name pilot that motivated the full study: sample-path fan charts (a clean
  "Kronos gets it right" and a clean "Kronos misses the earnings gap" case),
  and a vol-targeting / Sharpe robustness check. Superseded by the notebook
  above wherever the two differ; kept for the go/no-go reasoning and the plots.

Both notebooks are executed and self-contained — outputs are baked in, no need
to re-run anything to see the result.

## Methodology

- **Model:** `NeoQuasar/Kronos-small` (24.7M params), used strictly as a frozen
  forecaster — no training or fine-tuning.
- **The one piece of Kronos code in this repo** ([`src/kronos.py`](src/kronos.py)):
  `KronosPredictor.predict(sample_count=N)` runs N sampled autoregressive paths
  and **averages them** before returning — a single smoothed line with zero
  dispersion. `sample_paths()` is the identical batched inference, returning
  the array *before* the mean, so the full predictive distribution (used for
  every forecast, quantile, and calibration check here) is preserved.
- **Data:** Yahoo's chart API via `curl_cffi` (browser-TLS impersonation),
  daily and hourly OHLCV, 30 large-caps selected by a liquidity screen
  (`research/universe_screen.py`: trailing-year median dollar volume
  ≥ $300M/day and worst day ≥ $50M) and hand-balanced across sectors and
  volatility levels.
- **Out-of-sample discipline:** Kronos-small's pretraining ends ~June 2024
  (Shi et al., arXiv:2508.02739); every forecast origin here is July 2024 or
  later, so results are not measuring memorisation.
- **Target:** next-session realised volatility (Garman–Klass, intraday-only —
  the overnight gap is excluded as a separate risk).
- **Baselines:** EWMA, rolling-window, HAR-RV (Corsi), GARCH(1,1)/GJR — all
  refit walk-forward, using only information available at each forecast
  origin.
- **Tests:** QLIKE loss, Mincer–Zarnowitz regression (HAC standard errors),
  Diebold–Mariano, Hansen–Lunde–Nason Model Confidence Set, bootstrap CIs on
  incremental R², exact rank-PIT and CRPS for calibration.

## Reproducing

```bash
scripts/setup.sh      # venv + deps + vendored Kronos + weights
.venv/bin/python research/probe_hourly_rv/probe_hourly_rv.py   # the pilot
```

The full 30-name study that produced `02_vol_forecast_eval.ipynb` used a larger
harness (data pipeline, walk-forward baseline fitting, the evaluation harness
itself) that isn't included here — this repo keeps the result and the
methodology, not the engineering scaffolding. The notebook's code cells and
their outputs are the complete, exact record of what was computed.

## Layout

```
README.md                        this file — the finding
notebooks/
  02_vol_forecast_eval.ipynb      the primary result (executed)
  probe_findings.ipynb            the pilot (executed)
research/
  universe_screen.py / .csv       the 30-name liquidity + sector screen
  probe_hourly_rv/                the pilot study — script, data, README
src/kronos.py                     the one piece of Kronos code we wrote
scripts/setup.sh                  environment setup
```
