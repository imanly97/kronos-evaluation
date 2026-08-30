# Forecast Desk

An agentic research loop wrapped around a probabilistic price forecaster.

**Kronos** (a small foundation model over tokenised K-lines) produces a return
distribution per name per day. That distribution is a *number*, not a view. This
project is the loop a junior analyst runs around such a forecast:

- **Pre-market** — a **triage agent** ranks ~40 large-caps by how unusual today's
  forecast is *for that name*, investigates the top candidates (setup, analog
  base rate, correlated-cluster read, desk-memory track record), and briefs the
  few worth a human's morning. A **groundedness critic** signs off every number
  in every brief.
- **Post-close** — a **post-mortem agent** grades what settled and classifies
  *why* each watchlist name and each tail surprise happened:
  `within-expected-dispersion | catalyst:<type> | regime-move | data-issue | unexplained`.
- The post-mortem writes **desk memory** (append-only episodes + a derived
  per-(ticker, setup) stat table). The next morning's triage reads it back — so
  the loop learns this desk's track record with the model **without ever
  retraining the model**.

The forecasting pipeline is deterministic infrastructure — a scheduled job, not
an agent. The agentic work is the layer on top: *what is worth looking at* and
*what did we just learn*.

**No trades, no PnL, no alpha claim.** See [`PLAN.md`](PLAN.md) for the full
rationale, scope, and non-goals.

---

## Quick start

```bash
scripts/setup.sh                      # venv + deps + vendored Kronos + weights
cp .env.example .env                  # add ANTHROPIC_API_KEY (mock mode without)

# 1. build the deterministic forecast cache for the replay window
.venv/bin/python -m src.forecasts --start 2025-05-01 --end 2025-08-01

# 2. replay the whole loop day by day (memory accumulates)
.venv/bin/python -m src.run_daily replay --skip-forecast

# 3. read the evaluation
.venv/bin/python -m eval.evaluate
```

A single live session is two cron jobs:

```bash
.venv/bin/python -m src.run_daily premarket --asof 2025-05-29    # forecasts + triage
.venv/bin/python -m src.run_daily postclose --asof 2025-05-28    # grade + post-mortem
```

Outputs land in `log/` (`forecasts.jsonl`, `watchlist.jsonl`, `actuals.jsonl`,
`brief_<asof>.md`) and `memory/` (`episodes.jsonl`, `stats.json`), all
append-only and hash-chained — git history of those files is the tamper-evidence.

---

## Layout

```
PLAN.md                     the research plan (decisions, scope, evaluation)
src/
  config.py                 universe, paths, model + loop params — one file
  data.py                   prices + strict no-lookahead (Yahoo -> Stooq -> cache)
  kronos_infer.py           sample_paths(): Kronos WITH dispersion restored
  forecasts.py              deterministic Kronos batch -> immutable forecast cache
  setups.py                 technical features + controlled-vocab label + analogs
  features.py               cross-sectional ranking, strength-z, correlated cluster
  grade.py                  actual close -> quantile position, coverage, direction
  triage.py                 the pre-market triage agent (LangGraph)
  critic.py                 groundedness critic (deterministic + LLM passes)
  postmortem.py             the post-close post-mortem agent (LangGraph)
  memory.py                 desk memory: episodes + derived stats + recall()
  store.py                  append-only JSONL + hash chain + SQLite mirror
  llm.py                    shared Anthropic client + JSON/text helpers
  run_daily.py              premarket | postclose | replay
  mcp_server.py             the watchlist + why-flagged tools over MCP
eval/
  build_catalysts.py        scheduled-event calendar (Nasdaq + curated macro)
  evaluate.py               the metrics harness
notebooks/
  walkthrough.ipynb         one asof end to end
  loop_replay.ipynb         N days replayed; the memory ablation
```

## The one non-obvious thing

`KronosPredictor.predict(sample_count=N)` runs N sampled paths and then
**averages them** (`np.mean(preds, axis=1)`) before returning — a smoothed mean
line with zero dispersion. `src/kronos_infer.py::_auto_regressive_paths` is the
identical batched inference returning the array *before* the mean. That is the
whole envelope, and the envelope is the whole point.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Offline only — no network, no LLM, no Kronos. Covers the hash chain, row
immutability, the no-lookahead rule, the setup vocabulary, and the grade-quantile
interpolation.
