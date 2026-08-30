# Forecast Desk — an agentic research loop around a price forecaster

**Status:** in build (pivot from the "narrative arbitrage" v1)
**Owner:** you · **Assist:** Claude · **Root:** `~/Desktop/foundation_model_desk/`
**Repo:** `github.com/imanly97/narrative_arbitrage` (rename to `foundation_model_desk` pending)

---

## 1. Thesis

> A probabilistic price forecaster (**Kronos**) produces a distribution per name
> per day. That distribution is a *number*, not a view. This project builds the
> **research loop a junior analyst runs around such a forecast**: every morning,
> triage the universe down to the few names worth a human's attention and brief
> each one; every evening, review what happened and write down *why* the model
> was right or wrong. The evening notes accumulate into a **desk memory** that
> the next morning's triage reads back — so the loop learns this desk's track
> record with the model **without ever retraining the model**.

The forecasting pipeline is deterministic infrastructure — a scheduled job, not
an agent. The agentic work is the layer on top: *what is worth looking at*, and
*what did we just learn*. That split — and being able to defend it — is the point.

---

## 2. Scope & non-goals

**In scope:** deterministic Kronos batch + forecast cache; cross-sectional and
per-name-historical features; a **triage agent** that produces a briefed
watchlist; a **groundedness critic**; a **post-mortem agent** that classifies
each day's outcomes and writes structured memory; a **desk-memory store** with
key-based recall; the feedback edge (triage reads memory); an evaluation harness;
an MCP server; a blog post on the loop and the eval methodology.

**Non-goals:** trades, position sizing, PnL/returns, transaction costs, intraday
execution, any claim of tradable alpha. Also **not** retraining, fine-tuning, or
recalibrating Kronos — a real desk does that rarely and with heavy justification,
never on a nightly cadence. The model is a fixed input.

---

## 3. Decisions — LOCKED

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Primary mode | **Historical replay** of the loop over a chosen window (e.g. last 3 months, day by day) | Results now, fully reproducible; the memory accumulates and the ablation runs offline. Live daily deployment is optional, not load-bearing. |
| D2 | Forecaster | **Kronos-small**, 200 sampled paths, **120-bar lookback** | v1 finding: long daily lookbacks destabilise the sampler for names near range extremes. 120 bars is the stable operating point. Carried over. |
| D3 | Horizon | `pred_len = 3`, graded on **bar 1** (the `asof` bar), close-to-close | Carried over from v1. |
| D4 | Universe | **~40 liquid US large-caps** for the loop (dev on the v1 set of 12) | Triage needs enough names to be a real filter. ~40 × ~10s/name on MPS ≈ 7 min/day. |
| D5 | Prices | **yfinance/Yahoo chart API via curl_cffi → Stooq → cache** | Carried over from v1 (the plain requests UA gets 429'd). |
| D6 | Orchestration | **LangGraph** for the triage, critic, and post-mortem graphs | Real branching and loops, not linear chains. |
| D7 | LLM | **Anthropic API (Claude)**, key + optional workspace-id from env | Carried over. `.env` is gitignored. |
| D8 | Memory | **Append-only JSONL episodes + a derived per-(ticker, setup) stats table**, deterministic key-based recall | Inspectable; no embedding dependency; the recall is evaluable. Semantic recall is a v1.5 stretch. |
| D9 | Store | **Append-only JSONL + SHA-256 hash chain + SQLite mirror** | Carried over from v1 `store.py`. |
| D10 | Setup vocabulary | Small controlled list (momentum-breakout, range-bound, post-gap-drift, vol-expansion, mean-reversion-candidate, trend-continuation, quiet) | The shared key between a brief and a memory entry. |

---

## 4. The loop

```
                    ┌───────────────── DESK MEMORY (JSONL episodes + stats) ─────────────────┐
                    │                                                                       │
   PRE-MARKET       ▼                                                     POST-CLOSE         │
  ┌───────────────────────────┐                                    ┌───────────────────────────┐
  │ build_panel   (det.)      │                                    │ load_actuals  (det.)      │
  │ run_forecasts (det.)      │                                    │ grade_forecasts (det.)    │
  │ compute_features (det.)   │                                    │                           │
  │      │                    │                                    │      │                    │
  │      ▼                    │                                    │      ▼                    │
  │ TRIAGE agent (LangGraph)  │   ──► watchlist.jsonl  ──►  humans  │ POST-MORTEM agent         │
  │  loop: pick candidate →   │        + brief.md          read it  │  focus: watchlist names + │
  │   describe setup →        │                                    │   tail surprises          │
  │   find analogs →          │                                    │  classify each miss:      │
  │   recall memory →         │                                    │   within dispersion? or   │
  │   keep/drop → write brief │                                    │   identifiable cause?     │
  │      │                    │                                    │      │                    │
  │      ▼                    │                                    │      ▼                    │
  │ CRITIC loop: every number │                                    │ write memory episodes ────┘
  │  matches a computed value │                                    │ update (ticker,setup) stats
  └───────────────────────────┘                                    └───────────────────────────┘
```

**Deterministic (no LLM):** `build_panel`, `run_forecasts`, `compute_features`,
`load_actuals`, `grade_forecasts`, memory-stats rollups. These are cron jobs.

**Agentic:** triage, critic, post-mortem. Each is a LangGraph graph with genuine
state and conditional branching — the sequence of steps depends on what the
prior step found.

---

## 5. Agentic components

### 5.1 Triage agent
**Goal:** from ~40 forecasts, select the 3–7 worth a human's morning and brief each.

**Tools:**
- `rank_forecasts()` → table: median, P(up), dispersion, strength (|median|/dispersion), and each name's strength **z-scored against its own trailing year** (is today unusual *for this name*).
- `describe_setup(ticker)` → deterministic features (trend slope, distance from N-day high/low, realized vol vs trailing, recent gap, range position) + a controlled-vocab label (D10).
- `find_analogs(ticker, setup)` → past dates for this name in the same setup and their forward close-to-close outcomes.
- `correlated_cluster(ticker)` → are correlated names all pointing the same way (macro/sector) or is this idiosyncratic?
- `recall_memory(ticker, setup)` → the desk's prior episodes and hit rate for this exact situation.

**Loop / state:** candidates considered, findings, briefs drafted. Pick a
candidate → investigate → decide keep/drop → if keep, draft brief → next.
Terminate on a per-run token budget or when the ranked candidates are exhausted.

**Output:** `watchlist.jsonl` (structured) + `brief_<asof>.md` (PM-readable). Each
brief: why flagged, the setup, the analog base rate, the correlated-cluster
read, the desk-memory track record, and a one-line confidence caveat.

### 5.2 Groundedness critic
Writer/critic pair. The critic extracts every quantitative claim from each brief
and checks it against the computed feature/forecast values; flags unsupported
numbers, overclaims, and missing caveats; sends back for revision until clean.
Target: **100% of numeric claims traceable to a computed value.**

### 5.3 Post-mortem agent
**Goal:** for the watchlist names and the day's biggest surprises (actual far in
the forecast tail), classify *why*.

**Tools:** `quantile_of_actual(ticker)` (where did the close land in Kronos's
distribution), `sector_move(asof)` (did the whole cluster move — regime vs
idiosyncratic), `known_catalyst(ticker, asof)` (earnings-calendar / scheduled
macro lookup — a curated/queried set, **not** sentiment), `data_sanity(ticker)`.

**Classification per name:** `within-expected-dispersion` | `catalyst:<type>` |
`regime-move` | `data-issue` | `unexplained`.

**Output:** memory episodes: `{asof, ticker, setup, forecast_median, actual,
actual_quantile, classification, note}`; and updated per-(ticker, setup) rolling
stats: `n`, `hit_rate`, `mean_signed_error`, `mean_abs_error`.

### 5.4 Desk memory
`memory/episodes.jsonl` (append-only, hash-chained) + `memory/stats.json`
(derived, rebuildable). Recall is deterministic: `recall(ticker, setup)` returns
the episodes and the aggregate stat line. This is what closes the loop.

---

## 6. Deterministic core

| Module | Does |
|--------|------|
| `data.py` | no-lookahead daily bars, Yahoo→Stooq→cache *(carried over from v1, unchanged)* |
| `kronos_infer.py` | `sample_paths()` / `forecast()` — Kronos with dispersion restored *(carried over, ~unchanged)* |
| `features.py` | cross-sectional ranking, per-name strength z-score, correlated clusters |
| `setups.py` | technical-feature extraction + controlled-vocab setup label |
| `forecasts.py` | batch Kronos over the universe → forecast cache |
| `grade.py` | actual close → quantile position, envelope coverage, direction match |
| `store.py` | append-only JSONL + hash chain + SQLite mirror *(carried over, new schema)* |
| `llm.py` | shared Anthropic client (workspace-id aware) *(carried over)* |

---

## 7. Evaluation

Chosen partly *because* it is evaluable with checks you can reason about (unlike
conformal-coverage math).

**Triage agent**
- **Groundedness** — % of numeric claims in briefs that match a computed value. Hard target 100%.
- **Eventfulness** — flagged names vs unflagged: distribution of |actual return|, realized range, and |actual − forecast median| in quantile terms. Flagged should skew eventful.
- **Human agreement** — for ~20 sampled days you pick your own top-5 from the raw forecast table; measure overlap with the agent.
- **Ablation** — run triage with memory **on vs off** over a held-out stretch; compare eventfulness and human-agreement.

**Post-mortem agent**
- **Catalyst recall** — on known earnings/Fed days, does it name the catalyst as the cause? (planted-label test)
- **Dispersion self-consistency** — when it says "within expected dispersion," the actual should sit in the 50–80% interval; when it says "something happened," in the tail. Check the correspondence over many days.
- **Groundedness** — as above.

**Memory**
- **Base-rate validity** — does a setup's claimed hit rate actually predict forward outcomes on later, unseen dates?

---

## 8. Persistence (schema sketch)

```
forecasts(  asof, ticker, gen_ts, price_source, context_to, prev_close,
            q05,q25,q50,q75,q95, p_up, std, strength, strength_z )   -- immutable
watchlist(  asof, ticker, rank, setup, reason, analog_base_rate,
            cluster_read, memory_line, caveat, brief_md )            -- immutable, per run
actuals(    asof, ticker, eval_ts, prev_close, actual_close, ret,
            actual_quantile, inside_envelope, dir_match )            -- next day
memory/episodes(  asof, ticker, setup, forecast_median, actual,
                  actual_quantile, classification, note )            -- append-only, chained
memory/stats.json  ->  { "<ticker>|<setup>": {n, hit_rate, mean_signed_err, mean_abs_err} }
```

---

## 9. Repo layout

```
~/Desktop/foundation_model_desk/
  PLAN.md
  src/
    config.py         data.py       kronos_infer.py    llm.py        # carried over
    features.py        setups.py     forecasts.py       grade.py      # new deterministic
    triage.py          critic.py     postmortem.py      memory.py     # new agents
    store.py           run_daily.py  mcp_server.py                    # adapted / glue
  notebooks/
    walkthrough.ipynb            # one asof: forecast -> triage -> brief -> grade -> post-mortem
    loop_replay.ipynb            # N days replayed; memory accumulating; the ablation
  eval/
    build_catalysts.py         # scheduled-event calendar (Nasdaq earnings + curated macro)
    evaluate.py                # the metrics harness (§7)
    human_picks.json           # your own top-5 per day, for the human-agreement metric
  cache/prices/     log/     memory/     hf_cache/     vendor_kronos/     .venv/
  scheduling/       .github/workflows/     scripts/setup.sh
  blog/post_loop.md
```

---

## 10. What carries over from v1

| Keep as-is | Adapt | Drop |
|---|---|---|
| `data.py`, `kronos_infer.py`, `llm.py` | `config.py` (drop divergence/narrative knobs; add triage/memory/universe) | `narrative.py` |
| `scripts/setup.sh`, `.env`, `.gitignore`, `requirements.txt` | `store.py` (machinery kept, schema replaced) | `headlines_agent.py` |
| venv, vendored Kronos, weights | `evaluate.py` (Wilson CI + report skeleton kept, metrics new) | `divergence.py` |
| the v1 findings (curl_cffi Yahoo, MPS `.eval()`, sampler vendoring, 120-bar lookback) | `.github/workflows/`, `scheduling/` (same skeleton, new commands) | `case_studies/` + the 3 case notebooks |
| hash-chain + no-lookahead tests | `mcp_server.py` (new tools) | `blog/post1_build.md` |

Every line of the deterministic spine and all the hard-won infra survives. The
LLM layer changes — and that layer was always the part that would.

---

## 11. Ops

- **Replay** is the default: `run_daily replay` walks the window and the memory
  accumulates exactly as it would live.
- **Live** (optional): two scheduled jobs — pre-market triage, post-close
  post-mortem — via GitHub Actions (primary) or launchd (backup). Commit `log/`
  and `memory/` back each run; git history is the tamper-evidence.
- **Contamination — Kronos:** Kronos-small's pretraining data ends **~June 2024**
  (Shi et al., *Kronos*, arXiv:2508.02739; test period begins July 2024). The
  replay window (May–Aug 2025) is **~11 months out of sample** — the forecasts
  are genuine predictions, not lookups of known price paths. The `src/baseline.py`
  numbers confirm the model behaves like a real forecaster, not an oracle
  (unbiased median, *no* directional edge, overconfident intervals).
- **Contamination — agents:** the post-mortem *may* know historical catalysts —
  fine and helpful, since its job is to identify them and we score whether it
  does. The triage agent makes no market prediction (it selects and describes),
  so outcome-knowledge doesn't bias it. Historical replay is a legitimate test
  here, unlike v1.

---

## 12. Timeline

| Phase | Output |
|-------|--------|
| 1 | Deterministic core: `forecasts.py`, `features.py`, `setups.py`, `grade.py`; forecast cache for a replay window; new `store.py` schema |
| 2 | `triage.py` + tools + `critic.py`; `walkthrough.ipynb` for one `asof` |
| 3 | `postmortem.py` + `memory.py`; close the loop (triage reads memory) |
| 4 | `eval/` harness + `loop_replay.ipynb` (the ablation) |
| 5 | `mcp_server.py`; blog draft |
| 6 | (optional) launchd / GitHub Actions for live daily |

---

## 13. JD alignment (Balyasny — Applied AI Scientist)

- **Agentic workflows / research automation** → the triage + post-mortem loop automates the analyst's daily forecast-review cycle.
- **Agentic frameworks (LangGraph)** → three graphs with real state, branching, and loops.
- **Data pipelines** → the deterministic panel + forecast cache + grading.
- **Model-provider APIs (Anthropic)** → the agents run on Claude.
- **MCP for tool/data integration** → the watchlist and `why_did_we_flag(...)` exposed as MCP tools.
- **Traceable reasoning + evaluation** → the critic loop; every claim tied to a number; a real eval harness with planted cases and an ablation.
- **Explaining trade-offs to PMs** → the morning brief *is* the PM artifact.
- **Bias to shipping** → the loop runs (replay or live) and produces dated, accumulating artifacts.
- **Judgment about where agents belong** → the forecaster is deliberately *not* an agent, and the plan says why.

---

## 14. Risks

- **"Interesting" is fuzzy** → pin it to measurable proxies (strength, strength-z, cluster agreement, analog dispersion) and let the agent *rank and explain* within that, not invent criteria. Eventfulness eval keeps it honest.
- **Memory adds noise, not signal** → the ablation is designed to catch exactly this; if memory-off wins, that's a reportable finding.
- **Setup labels too coarse/fine** → start with 7, adjust once the analog and memory retrieval have real data behind them.
- **Kronos systematically biased** (mild positive lean at 120 bars) → grade in quantile terms, not just direction; report the bias.
- **Replay window regime-specific** → pick a window spanning at least one vol spike; note it.
- **Agent cost** → deterministic core does the heavy compute; LLM only touches ~7 names/day for triage + the surprises for post-mortem. Budget-capped per run.

---

## 15. Open items — RESOLVED 2026-08-29

1. **Replay window** — `2025-05-01 → 2025-08-01` (spans the tariff-truce melt-up and the summer chop). LOCKED as D1/§3.
2. **Universe size** — expand to **~40 names now**; dev on an 8-name subset (`DEV_UNIVERSE`). LOCKED as D4.
3. **Repo rename** — folder → `foundation_model_desk` (done). GitHub repo rename to match is pending (auto-redirects; not blocking).
4. **Live deployment** — build the scheduled jobs this round (`.github/workflows/`, `scheduling/`), replay stays the default mode.
