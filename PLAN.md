# Daily Narrative Arbitrage — Project Plan

**Status:** draft for sign-off · **Owner:** you · **Assist:** Claude
**Project root:** `~/Desktop/narrative arb/`

---

## 1. Thesis (the honest research claim)

> A daily-frequency pilot using free retail data. Each pre-market open we generate
> a signal from (a) **Kronos**'s probabilistic price forecast, which sees only price
> structure, and (b) an **LLM narrative score** built from overnight headlines, which
> is blind to price. We measure whether — and *when* — these two disagree, and which
> one better anticipates the day's actual close-to-close move. **No execution, no PnL,
> no trades.** Just tracked signals, evaluated at the close, logged immutably for a month.

This is deliberately a *pilot*, not a statistical proof. Daily frequency gives one
observation per ticker per day; a month is ~20 trading days × 12 tickers ≈ 240 signal-days.
Enough for calibration eyeballing and illustrative divergence cases with honest confidence
intervals — not enough for a strong alpha claim, and we won't pretend otherwise.

---

## 2. Scope & non-goals

**In scope:** signal generation, distributional forecasting, narrative scoring, a
divergence verdict, immutable logging, close-of-day evaluation, and two blog posts.

**Non-goals (explicit):** placing trades, position sizing, PnL/backtest returns,
transaction-cost modeling, intraday execution, or any claim of tradable alpha.

---

## 3. Decisions — LOCKED

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Frequency | **Daily** | Kronos small/base context = 512 → ~2yr lookback; free daily data is the most reliable; overnight-headline → next-close horizon maps cleanly to one daily bar. |
| D2 | Model | **Kronos-small** nightly (base for spot-checks) | 24.7M params runs on the MacBook's **MPS** (auto-detected); base for occasional deeper looks. |
| D3 | Horizon | **pred_len = 3**, graded on **bar 1** | First predicted bar = `asof`; we grade its close-to-close move same day. Bars 2–3 shown for context only. |
| D4 | Prices | **yfinance** (API call) → **Stooq** fallback → cache | yfinance is primary as requested; Stooq covers Yahoo's periodic outages; CSV cache is inspectable. |
| D5 | Historical headlines | **Manual curation** (cited) | yfinance `.news` is current-only and flaky; past headlines can't be scraped with trustworthy timestamps. |
| D6 | Live headlines | **Nightly LangGraph agent** | Real tool-use: source selection, dedupe, materiality judgment, graceful no-news. |
| D7 | LLM | **Anthropic API (Claude)**, key from env | You supply `ANTHROPIC_API_KEY`; never committed. |
| D8 | Orchestration | **LangGraph** state machine | Directly matches the JD's agentic-framework requirement; genuine branching/state, not cosmetic. |
| D9 | Store | **SQLite**, append-only, timestamped | Signals locked at generation; evaluation joined later by (ticker, date). |
| D10 | Schedule | **launchd** (macOS) | Native; two jobs (pre-market signal, post-close eval). Laptop-sleep caveat handled in §10. |

---

## 4. The one thing that would have silently broken it

Kronos's `predict(..., sample_count=N)` runs N sampled paths in one batched pass and then
**averages them** (`np.mean(..., axis=1)` in `auto_regressive_inference`) before returning.
So the vanilla call yields a smoothed *mean* path and **zero dispersion** — no 5th–95th
envelope, which is the entire point of the project.

**Fix (locked):** vendor a ~20-line `sample_paths()` that runs the identical batched
inference but returns the array *before* the mean → shape `(sample_count, horizon, features)`.
One pass, full distribution, code we own. This lives in `src/kronos_infer.py`.

---

## 5. Architecture

Nightly run is a **LangGraph graph**, not a linear script. Guiding principle: *use an agent
only where there is genuine reasoning/tool-selection; keep everything else deterministic* —
and say so in the post (that restraint is itself the signal BAM broadcasts).

| Stage | Agentic? | What it does |
|-------|----------|--------------|
| `load_prices` | Deterministic | yfinance→Stooq, cache, `asof` no-lookahead rule |
| `run_kronos` | Deterministic | `sample_paths()` → per-step quantiles (5/25/50/75/95) |
| `headlines_agent` (per ticker) | **Agent** | Source selection, fetch, dedupe, materiality, no-news handling — price-blind |
| `score_narrative` | Structured LLM | direction ∈ {bear/neutral/bull} + conviction 1–10 + rationale — price-blind |
| `assess_divergence` | **Agent** | Reasons over Kronos quantiles + narrative; emits a PM-readable brief — grounded in the numbers |
| `persist_signals` | Deterministic | Lock + timestamp to SQLite |

**Second graph (post-close, next day):** `evaluate` joins locked signals with the actual
close → coverage, direction hit, divergence outcome.

**Conditional edges:** yfinance fail → Stooq → else drop ticker & log; no material news →
neutral narrative but still score structure (a structure-only signal with no catalyst is
itself an interesting row).

**MCP (differentiator):** expose `get_kronos_distribution(ticker)` and `get_headlines(ticker)`
as MCP tools so the same capabilities are callable by any MCP client — the post's "a PM's
assistant could query this directly" framing. Slots into the headline/divergence chunk.

---

## 6. Divergence definition — v1 (finalize at chunk 4)

- **Narrative signal:** `S_N = direction × conviction`, direction ∈ {−1, 0, +1}, conviction ∈ [1,10]. Price-blind.
- **Structural signal (from Kronos next-day close-return distribution `r`):**
  - direction = `sign(median(r))`
  - strength = `median(r) / std(r)` (median move in units of the forecast's own dispersion), plus `P(r > 0)`.
- **Divergence types:**
  - *Directional:* `sign(S_N) ≠ sign(median(r))`.
  - *Magnitude:* narrative high-conviction, but the narrative-implied direction sits in a low-probability region of Kronos's distribution (structure thinks the expected move is unlikely).
- **Adjudication (at close):** on divergence rows, which side matched the sign (and rough magnitude) of the actual close-to-close return?

---

## 7. Evaluation metrics

- **Calibration / coverage:** fraction of actual closes landing inside the 5–95 envelope (well-calibrated ⇒ ~90%). Report per-ticker and pooled, with the caveat that n is small.
- **Directional hit rate:** did `sign(median(r))` match the actual return sign? Baseline 50%. Same for narrative direction.
- **Divergence-conditioned:** on rows where narrative and structure disagreed, win rate of each side. This is the headline research question.
- **Data quality:** scheduled-vs-actual run coverage (see §10) reported honestly.

---

## 8. Persistence (SQLite schema sketch)

```
signals(   asof DATE, ticker TEXT, gen_ts TIMESTAMP,
           kronos_q05, kronos_q50, kronos_q95, kronos_p_up, kronos_std,
           narr_dir INT, narr_conviction INT, narr_rationale TEXT,
           divergence_type TEXT, divergence_verdict TEXT, brief TEXT,
           PRIMARY KEY (asof, ticker) )        -- locked, never updated

actuals(   asof DATE, ticker TEXT, prev_close, actual_close, ret,
           inside_envelope BOOL, dir_match_struct BOOL, dir_match_narr BOOL,
           PRIMARY KEY (asof, ticker) )        -- written next day by evaluate
```

Signals are immutable once written; evaluation only ever inserts into `actuals`.

---

## 9. Repo layout (under project root)

```
~/Desktop/narrative arb/
  PLAN.md
  case_studies/
    nvda_2025-05-29.md         # ← exemplar (this batch)
  src/
    data.py                    # ✓ built & offline-tested
    kronos_infer.py            # chunk 2: sample_paths + quantiles + fan chart
    headlines_agent.py         # chunk 3: LangGraph headline sub-agent
    narrative.py               # chunk 3b: structured scorer (Claude)
    divergence.py              # chunk 4: grounded divergence agent
    graph.py                   # chunk 5: LangGraph wiring
    store.py                   # SQLite log + eval join
    evaluate.py                # chunk 6: coverage / direction / divergence
    run_nightly.py             # scheduler entrypoint
    mcp_server.py              # optional: expose tools over MCP
  cache/prices/                # inspectable CSVs
  narb.db                      # SQLite
  scheduling/
    com.narrativearb.signal.plist
    com.narrativearb.evaluate.plist
  blog/
    post1_build.md
    post2_results.md
```

---

## 10. Ops for a month-long unattended run

- **Scheduling:** `launchd` with two `StartCalendarInterval` jobs — signal ~08:45 ET, evaluate ~16:15 ET.
- **Laptop-sleep reality:** if the lid is shut at trigger time, the job is missed. Mitigations: keep plugged in and use `caffeinate`, or `pmset schedule wake`, or accept misses and **log** them. The follow-up post reports run coverage as a data-quality section, not an embarrassment. If misses get bad, lift the nightly job to a small cloud box.
- **Secrets:** `.env` + `python-dotenv`; `ANTHROPIC_API_KEY` from env; `.gitignore` the `.env`, `cache/`, and `narb.db`.

---

## 11. Timeline

| Phase | When | Output |
|-------|------|--------|
| 0 | now | This plan + exemplar case study (sign-off) |
| 1 | Sat | `kronos_infer.py` (sampler, quantiles, first real fan chart), `store.py`; verify on 1–2 tickers |
| 2 | Sun | Run 3 case studies (manual headlines); draft **post 1**: architecture + case studies |
| 3 | Sun night → | Stand up the LangGraph nightly graph + launchd; **go live ~1 month** |
| 4 | after ~20 trading days | `evaluate` results; draft **post 2**: the follow-up |

**Post 1** publishes after Phase 2 ("live tracking starts today"). **Post 2** is the results follow-up.

---

## 12. Blog structure (chunked, so it's never a black box)

1. **The claim & why daily** — framing, non-goals.
2. **Data** — yfinance-as-API, the `asof` no-lookahead rule, cache, Stooq fallback (show the summary table).
3. **Kronos & trajectories** — the tokenizer/AR idea, the averaging gotcha, sampled paths (show raw trajectories).
4. **Distribution** — fan charts, per-step quantiles, a coverage sanity check.
5. **Narrative** — the price-blind scorer, prompt design, an example score + rationale.
6. **Divergence** — the agent, the v1 definition, a worked verdict.
7. **Evaluation** — coverage, direction, divergence-conditioned; honest CIs.
8. **Architecture** — the LangGraph graph, agent-vs-deterministic restraint, MCP.
9. **Live tracking begins** — what post 2 will answer.

---

## 13. JD alignment (Balyasny — Applied AI Scientist)

- *Agentic workflows / research automation* → the LangGraph nightly graph producing per-ticker research briefs.
- *Data pipelines* → `data.py` + `store.py` + scheduled runs.
- *Agentic frameworks (LangGraph)* → `graph.py`.
- *Model-provider APIs (Anthropic)* → narrative + divergence via Claude.
- *MCP for tool/data integration* → `mcp_server.py`.
- *Traceable reasoning + evaluation* → grounded briefs + the eval harness + the immutable log.
- *Explaining trade-offs to PMs* → the brief format + the posts themselves.
- *Bias to shipping* → it runs unattended for a month and produces a results post.

---

## 14. Open items (need your call)

1. **Divergence v1** — accept §6 as the starting definition, or adjust before chunk 4?
2. **Case-study set** — exemplar is NVDA earnings (narrative's easy case). Proposed partners: one **no-catalyst** day (does narrative just add noise?) and one **narrative head-fake** (headlines screamed, stock did the opposite). OK to hunt for those two?
3. **MCP** — build the MCP server this round, or defer to post-1-ship as a "future work" hook?

---

## 15. Risks

- **Small n** — mitigated by framing as a pilot + reporting CIs.
- **Scheduled catalysts favor narrative trivially** — mitigated by including non-catalyst and head-fake cases.
- **yfinance breakage mid-run** — mitigated by Stooq fallback + cache.
- **Laptop misses runs** — mitigated by logging coverage; cloud lift if severe.
- **LLM narrative leakage of price** — mitigated by strict price-blind prompt + logging exactly what the agent saw.

---

## 16. Build session 2026-08-29 — decisions & refinements

**Open items from §14, resolved:**

| # | Resolution |
|---|-----------|
| 14.1 | Divergence v1 **accepted** with concrete thresholds now in `src/config.py`: directional needs conviction ≥ 4; magnitude needs conviction ≥ 7 **and** `P(return in narrative's direction) < 0.30` under Kronos. |
| 14.2 | 3 case studies (NVDA exemplar + one no-catalyst + one head-fake). Headlines fed **verbatim, not scrubbed** — see §5a. |
| 14.3 | **MCP built this round** — `src/mcp_server.py` exposes `get_kronos_distribution`, `get_headlines`, `get_narrative_score`, `get_signal`. |

**§5a — the price-blindness caveat (new, important).**
Scrubbing price mentions from headlines does *not* make a historical case study
truly price-blind: Claude may carry the realised outcome in its training data, and
that is undetectable. So we don't scrub. Instead every signal row carries
`llm_contaminated` = (`asof` ≤ knowledge cutoff). Case studies are **illustration**;
the **live run** (all dates after the cutoff) is the actual test. Stated plainly in
post 1.

**Ticker universe (LOCKED):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD,
NFLX, AVGO, JPM, XOM.

**Scheduling (LOCKED): GitHub Actions is primary**, launchd is a local backup.
Cloud cron removes the laptop-sleep risk class entirely; committing `log/` back
each run makes git history the tamper-evidence. `scheduling/*.plist` kept for
local use. Replaces most of §10's mitigation burden.

**Immutable log — refinement of §8.** Source of truth is now an append-only
**JSONL** (`log/signals.jsonl`, `log/actuals.jsonl`) with a per-row SHA-256 hash
chained off the previous row. `narb.db` is a disposable SQLite mirror rebuilt from
the JSONL (`python -m src.store sync`). A `--replace` re-lock appends a superseding
row; the queryable view takes last-per-key while the chain still covers every line.

**Evaluation timing — refinement of §7/§10.** Grade the **morning after** the
session (~09:45 ET), not 16:15 ET, so Yahoo's consolidated official close is
settled.

**Data sourcing note.** Yahoo now 429s the stdlib user agent; we hit the chart
API through `curl_cffi` (browser TLS impersonation) and adjust OHLC by
`adjclose/close`. Stooq fallback is best-effort (JS-challenged from some IPs).

**Reproducibility.** Kronos seed is `sha256(ticker|asof)` — stable across
processes (Python's builtin `hash()` is per-process salted). Verified identical
output across runs on MPS. `.eval()` on model+tokenizer is required on MPS (SDPA
rejects a non-zero `dropout_p`).

**Environment.** Project-local `.venv` (pinned `requirements.txt`), not the
anaconda base env — keeps `huggingface_hub==0.33.1` (Kronos's pin) isolated and
the user's base jupyter untouched. `scripts/setup.sh` is idempotent.

**Notebooks (new deliverables):**
- `notebooks/walkthrough.ipynb` — one ticker, one day, every stage explained in
  detail. Regenerable via `notebooks/_build_walkthrough.py`.
- `notebooks/case_nflx_2024-04-19.ipynb` — head-fake (both signals fooled).
- `notebooks/case_jpm_2025-06-11.ipynb` — no-catalyst day.
- Case notebooks regenerable via `notebooks/_build_case.py`.

**Kronos lookback — a finding, not a config choice.** Feeding the full ~400–512
daily bars destabilises the AR sampler for names trading near the high/low of the
window (normalised last price → multi-sigma → hard reversion; medians of −3 to
−8%, tails past −30%). **`LOOKBACK_BARS = 120`** (~6 months) is the stable
operating point across all 12 tickers; mid-range names were unaffected either way.
Reported in post 2's calibration section.

**Case-study results (run 2026-08-29, lookback 120, 200 paths):**

| case | actual | Kronos median | 5–95 | narrative | verdict | inside? |
|---|---|---|---|---|---|---|
| NVDA 2025-05-29 (catalyst) | +3.25% | +1.79% | [−0.97, +5.31] | bull/7 | aligned | yes |
| NFLX 2024-04-19 (head-fake) | **−9.09%** | +1.13% | [−1.59, +3.97] | bull/7 | aligned | **no** |
| JPM 2025-06-11 (no-catalyst) | −0.17% | +0.33% | [−1.48, +1.98] | neutral/1 | aligned | yes |

None of the three produced a divergence — on "obvious" days the two views tend to
agree. NFLX is a double miss (both bullish, stock gapped down through the
envelope floor): the scorer's own rationale named the disclosure red flag and
still weighted the beat higher.
