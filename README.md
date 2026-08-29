# Daily Narrative Arbitrage — pilot

Each pre-market open, build two independent views of the trading day and log where
they disagree:

- **Structure** — [Kronos](https://github.com/shiyu-coder/Kronos), a foundation
  model over tokenised K-lines, run as a **distribution** (200 sampled paths),
  seeing only price/volume history strictly before `asof`.
- **Narrative** — Claude scoring overnight headlines, **price-blind**.

Grade at the close: was the actual return inside Kronos's 5–95 envelope? Which
side called the direction? On divergence days, which side won?

**No trades, no PnL, no alpha claim.** It's a pilot — ~20 trading days × 12
tickers ≈ 240 signal-days, logged immutably, evaluated with honest confidence
intervals. Full rationale in [`PLAN.md`](PLAN.md).

---

## Quick start

```bash
scripts/setup.sh                       # venv + deps + Kronos weights
cp .env.example .env                   # add ANTHROPIC_API_KEY (optional; mock mode without)
.venv/bin/python -m src.graph --asof 2025-05-29 --tickers NVDA
```

Then open the notebooks:

```bash
.venv/bin/jupyter lab notebooks/
```

- **`walkthrough.ipynb`** — the whole pipeline for one ticker/day, every stage
  explained: no-lookahead cut, raw Kronos paths, fan chart, headline sub-agent
  trace, narrative score, divergence brief, immutable write, close-of-day grade.
- **`case_nflx_2024-04-19.ipynb`** — narrative head-fake: a strong earnings print,
  both signals bullish, stock −9%.
- **`case_jpm_2025-06-11.ipynb`** — a no-catalyst day: narrative correctly
  abstains.

Rebuild any of them with `python notebooks/_build_walkthrough.py` /
`python notebooks/_build_case.py all`.

---

## Layout

```
PLAN.md                     the research plan (decisions, scope, non-goals)
src/
  config.py                 universe, paths, model params, divergence thresholds
  data.py                   prices + strict no-lookahead rule (Yahoo→Stooq→cache)
  kronos_infer.py           sample_paths(): Kronos WITH dispersion + fan chart
  headlines_agent.py        LangGraph headline sub-agent (price-blind)
  narrative.py              structured price-blind scorer (Claude tool-use)
  divergence.py             deterministic classification + LLM PM brief
  graph.py                  the nightly signal LangGraph
  evaluate.py               post-close grading LangGraph + running report
  store.py                  append-only JSONL log w/ hash chain + SQLite mirror
  run_nightly.py            scheduler entrypoint (signal | evaluate)
  mcp_server.py             the two capabilities exposed as MCP tools
notebooks/walkthrough.ipynb one signal-day, end to end
case_studies/               worked examples (curated + cited headlines)
log/signals.jsonl           the immutable signal log (tracked; git = tamper-evidence)
scheduling/                 launchd plists (local backup scheduler)
.github/workflows/          GitHub Actions nightly (primary scheduler)
tests/                      offline tests for the deterministic pieces
```

## The one non-obvious thing

`KronosPredictor.predict(sample_count=N)` runs N sampled paths and then
**averages them** (`np.mean(preds, axis=1)`) before returning — a smoothed mean
line with zero dispersion. `src/kronos_infer.py::_auto_regressive_paths` is the
identical batched inference returning the array *before* the mean. That's the
whole envelope.

## Going live

1. Create a GitHub repo, push, add `ANTHROPIC_API_KEY` as an Actions secret.
2. The workflow in `.github/workflows/nightly.yml` runs `signal` ~08:45 ET and
   `evaluate` ~09:45 ET next morning, committing `log/` back each run. Git
   history of `log/signals.jsonl` is the immutable record.
3. Local alternative: `cp scheduling/*.plist ~/Library/LaunchAgents/ && launchctl load …`
   (subject to the laptop-sleep caveat in `PLAN.md §10`).

## Tests

```bash
.venv/bin/python -m pytest -q
```
