# Evaluating Kronos for equity price forecasting

**Status:** plan for sign-off · reset from the v1 "agentic desk" (archived at
[`docs/PLAN_v1_agentic_desk.md`](docs/PLAN_v1_agentic_desk.md); tags
`v0.2-agentic-loop`, `v0.3-probe`)

---

## 1. The question

[Kronos](https://github.com/shiyu-coder/Kronos) is an open-source foundation
model for financial candlesticks (K-lines), pretrained on ~12B bars from 45
exchanges. Does it have **genuine, baseline-beating forecasting skill on US
equities** — and *where*: which horizon, which frequency, which target (return
direction / realised volatility / distribution), which regime?

The deliverable is a **map** — a matrix of `(task × condition) → skill vs the
right baseline`, with the statistics to defend each cell — not a verdict.

We do **not** train, fine-tune, or adapt Kronos. It is a frozen input.

---

## 2. What we already know (the v0.3 probe)

From `research/probe_hourly_rv/` (8 names, 960 forecasts, out of sample):

| finding | detail |
|---|---|
| **no directional skill** (daily) | sign hit 49.5% vs 54.6% majority-class base rate |
| **modest realised-vol skill** (hourly) | +0.053 incremental log-RV R² over EWMA, bootstrap 95% CI [+0.028, +0.084]; within-name Spearman 0.29 vs EWMA 0.22 |
| edge concentration | high-vol names, turbulent regimes; ~nil on the quietest names |
| **miscalibrated on level** | Mincer–Zarnowitz slope 0.82, biased low; expanding-window affine recal fixes QLIKE 1.9 → 1.1 |
| overconfident intervals | daily 90% band covers 78%; PIT U-shaped |
| the failure mode | big misses cluster on price gaps — scheduled/unscheduled news the price-only model can't see (TSLA earnings, NVDA/DeepSeek) |
| contamination boundary | Kronos-small pretraining ends **~June 2024** (Shi et al., arXiv:2508.02739; test period starts July 2024) |

This plan turns the probe into a proper study: right baselines refit
walk-forward, real hypothesis tests, honest splits, more names, both frequencies.

---

## 3. Scope

**In:**
- Assets: ~30 US large-cap equities (liquid, ≥5y history, sector-spread)
- Frequencies: **daily** and **hourly**
- Targets: (1) realised volatility *(primary)*, (2) predictive-interval /
  distribution calibration, (3) return direction *(confirming null)*
- Baselines per target, all refit walk-forward
- The recalibration layer as a studied component
- Two mechanism deep-dives: *why* the RV skill; the calibration failure + fix

**Out (for now):**
- Crypto, FX, other asset classes
- Scheduled-catalyst error attribution → **follow-up study**
- Any strategy / backtest / economic-value / alpha claim → **revisit after the
  forecast-quality results are in**
- Fine-tuning or adapting Kronos
- 15-minute / tick frequency
- Any agentic / LLM layer

---

## 4. Data

- Source: Yahoo chart API via `curl_cffi` (browser-TLS impersonation — the plain
  UA gets 429'd), Stooq CSV fallback, on-disk cache. Carried over from v0.x
  `data.py`, extended to intraday.
- **Daily:** 2018-01 → present, split+dividend adjusted OHLCV.
- **Hourly:** 2023-10 → present (Yahoo's 730-day rolling limit), regular session
  only (`includePrePost=false`), ~7 bars/session.
- **Universe:** ~30 names picked in W1 — top US market cap, options-liquid, clean
  history, sector coverage. Frozen and committed once chosen (LOCKED).
- **No-lookahead:** every forecast's context ends strictly before its origin
  timestamp. Verified by test on every run.

---

## 5. Splits — controlling *our* overfitting (Kronos is frozen)

| window | dates | use |
|---|---|---|
| **Contaminated** | ≤ 2024-06-30 | **daily only** (hourly history barely predates the cutoff). Used only to *measure the contamination gap*; never a headline skill number. |
| **OOS-development** | 2024-07-01 → 2025-12-31 (~18 mo) | every methodology choice: lookback, RV estimator, recal fit, baseline specs, regime cuts, metrics. |
| **OOS-lockbox** | 2026-01-01 → 2026-08-31 (~8 mo) | touched **once**, at the end. Dev ≈ lockbox → robust; diverge → we fooled ourselves. |

- **Rolling-origin everywhere.** Every forecast is strictly causal.
- Recalibration layer and GARCH/HAR baselines refit on **expanding windows** —
  nothing peeks.
- The lockbox is a consistency check on the **vol** metrics (ample data). It is
  **not** enough for a direction/Sharpe CI — that would need years.

---

## 6. Targets & metrics

### 6.1 Realised volatility (primary)

- **Origin convention (LOCKED):** session-aligned. Origin = a session close;
  forecast covers the **next full session's intraday bars**. Excludes the
  overnight jump — cleaner target, matches "what will tomorrow be like." (The
  probe's mixed-hour origins were a wart; this fixes it.)
- **RV estimator:** range-based **Garman–Klass / Parkinson** as primary (uses
  each bar's OHLC → ~5× more efficient than close-to-close, and Kronos forecasts
  full OHLC so it applies to the sample paths too); close-to-close RV as a
  robustness check.
- **Horizons:** daily H ∈ {1, 5} sessions; hourly H ∈ {1, 2} sessions
  (≈ {7, 14} bars).
- **Kronos forecast:** median (and mean) over N sample paths of each path's
  forward RV. N-sensitivity checked (100 / 200 / 400 paths).
- **Baselines (walk-forward):** EWMA/RiskMetrics (λ chosen on dev), HAR-RV
  (Corsi), GARCH(1,1) and GJR-GARCH (`arch`), rolling historical vol, RV random
  walk.
- **Metrics:** QLIKE *(primary — robust to the noisy-target problem)*, MSE on
  log-RV, Mincer–Zarnowitz regression (intercept, slope, R²).
- **Tests:** Diebold–Mariano with HAC (Newey–West) SEs, Kronos vs each baseline;
  incremental-information regression (`log RV ~ log EWMA + log Kronos` — the
  probe's key result, done properly with clustered SEs); Hansen–Lunde–Nason
  **Model Confidence Set** across the zoo.
- **Cuts:** realised-vol quartile, name, sector, calendar year, in/out of sample.
- **Recalibration study:** fit `RV ~ a + b·KronosRV` expanding-window; report
  pre/post QLIKE and MZ; does the recalibrated forecast enter the MCS?

### 6.2 Interval / distribution calibration

- **PIT:** where does the actual land in Kronos's predictive distribution?
  Should be Uniform(0,1). Kolmogorov–Smirnov + Christoffersen coverage /
  independence tests.
- **Coverage:** does q05–q95 cover 90%? q25–q75 cover 50%? Per-quantile.
- **Pinball / quantile loss** at {.05,.25,.5,.75,.95} vs Gaussian-EWMA and
  empirical-residual baselines.
- **CRPS** of the full sample-path distribution vs baselines.
- **Conditional calibration:** does it degrade in high-vol regimes / at longer
  horizons? (probe says yes)

### 6.3 Direction (confirming null)

- Sign of the next-H return. Accuracy, log-loss, AUC vs random walk / AR(1) /
  historical drift. Pesaran–Timmermann directional-accuracy test.
- Expectation: no skill. Confirm cleanly; don't belabour.

---

## 7. Deep-dives

1. **Why does Kronos beat EWMA on RV?** Decompose the edge: faster
   mean-reversion after a spike (the probe's AMD case), volume information,
   candle-shape (range/body ratios), regime detection. Compare how Kronos's RV
   forecast moves vs EWMA/HAR around vol regime shifts. Input-feature ablations
   where the sampler allows.
2. **The calibration failure.** Characterise the bias — level, regime
   dependence, horizon dependence. The recal fix and its limits. Is the
   overconfidence a sampling artefact (more paths?) or intrinsic to the model?
3. *(light)* **The gap misses.** Descriptive only: what fraction of the worst-5%
   QLIKE observations fall on earnings days? Uses a calendar, but the full
   attribution + fix is the deferred follow-up.

---

## 8. Rigor checklist

- [ ] no-lookahead verified by test (`context.max < origin`, every run)
- [ ] all baselines walk-forward; no in-sample fitting anywhere
- [ ] recalibration expanding-window only
- [ ] DM tests with HAC SEs; MCS across models
- [ ] multiple-testing acknowledged / FDR-controlled across the `(task × cut)` grid
- [ ] dev results frozen & committed before the lockbox is touched
- [ ] seeds fixed; sample-path-count sensitivity reported
- [ ] contamination gap reported (daily)
- [ ] every headline number reproducible from `evaluate.py` + committed forecast store

---

## 9. Repo structure

```
PLAN.md
docs/PLAN_v1_agentic_desk.md        # the archived prior direction
src/
  kronos.py          # the dispersion-preserving sampler (from v0.x kronos_infer.py, trimmed)
  data.py            # no-lookahead loader — daily + hourly (extended from v0.x)
  config.py          # paths, universe, splits, model params — one file
  targets.py         # RV (Garman–Klass / close-to-close) + direction + quantile targets
  baselines.py       # EWMA, HAR-RV, GARCH, GJR, RW — walk-forward
  forecast.py        # run Kronos over a grid of origins -> immutable forecast store
  metrics.py         # QLIKE, MSE, pinball, CRPS, PIT, MZ, Diebold–Mariano, MCS
  recalibrate.py     # the expanding-window affine layer
  evaluate.py        # the harness: assemble the (task × cut) tables
research/
  probe_hourly_rv/                  # the v0.3 pilot — kept as-is
notebooks/
  01_data_and_targets.ipynb
  02_vol_forecast_eval.ipynb        # the headline
  03_calibration.ipynb
  04_direction_null.ipynb
  05_mechanism_why_rv.ipynb
report/
  findings.md
tests/
cache/  vendor_kronos/  hf_cache/  .venv/  scripts/setup.sh
```

---

## 10. Timeline (~3–4 weeks)

| week | output |
|---|---|
| 1 | data (daily + hourly, ~30 names), targets, baselines, no-lookahead tests. Reproduce the probe's hourly-RV result properly — session-aligned origins, DM tests. |
| 2 | full vol eval on **dev** — every cut, every baseline, MCS, the recalibration study. |
| 3 | calibration + direction. Mechanism deep-dive (why RV). |
| 4 | **lockbox** run (once), `report/findings.md`, notebook polish. |

---

## 11. Non-goals, restated

No alpha claim without a separate economic-value analysis (deferred). No
fine-tuning. No crypto. No agents. The output is a defensible answer to: *where
does Kronos forecast US-equity volatility better than the standard models, where
does it fail, and by how much.*
