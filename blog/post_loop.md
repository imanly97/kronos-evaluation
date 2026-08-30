# Building the research loop around a price forecaster

*Draft. Companion to the [Forecast Desk](https://github.com/imanly97/foundation_model_desk) repo.*

## The setup

A probabilistic price forecaster — here [Kronos](https://github.com/shiyu-coder/Kronos),
a small foundation model over tokenised candlesticks — gives you a return
distribution per name per day. Run it over 40 large-caps and every morning you
have 40 little fans of possible outcomes.

That is not a research product. It is an input. The actual daily work is: *which
of these 40 is worth a human's attention*, and, at the end of the day, *what did
we just learn about this model on this desk*. That work is a loop, and this post
is about building the loop as an agentic system while keeping the forecaster
itself as dumb, deterministic infrastructure.

## Where the agent belongs — and where it doesn't

The forecasting pipeline is a cron job: pull no-lookahead price history, run 200
sampled Kronos paths, reduce to quantiles, write an immutable row. No LLM touches
it. Making the forecaster an "agent" would add nondeterminism to the one part of
the system that most needs to be reproducible.

The agentic layer sits on top, in two graphs:

- **Triage** (pre-market): rank the 40 forecasts by how unusual today's is *for
  that name* (strength z-scored against its own trailing year), then walk the top
  candidates — describe the technical setup from a controlled vocabulary, pull
  the base rate of that setup's past instances, check whether correlated names
  agree, recall the desk's own track record — and decide keep or drop. Kept names
  get a brief.
- **Post-mortem** (post-close): grade what settled, and for the watchlist names
  and the day's tail surprises, classify *why*:
  `within-expected-dispersion | catalyst:<type> | regime-move | data-issue | unexplained`.

Each is a LangGraph graph with real branching — the next step depends on what the
last one found — not a linear chain.

## The feedback edge

The post-mortem writes **desk memory**: an append-only episode per classified
name, plus a derived per-(ticker, setup) stat table (n, hit rate, mean signed
error). The next morning's triage reads that table back. Over a replay the memory
accumulates and the loop learns the desk's track record with the model **without
retraining the model** — which is what a real desk does, because retraining a
forecaster nightly is not a thing.

## Keeping the agent honest

Two mechanisms:

1. **A groundedness critic.** Every number in every brief is traced — first
   deterministically (pull the numerals out of the prose, match each to a
   computed value within tolerance), then with an LLM pass for overclaims and
   missing caveats. Briefs loop through revision until clean. Target: 100% of
   numeric claims tied to a computed value.
2. **An evaluation harness built from checks you can reason about.** Eventfulness
   (do flagged names skew toward large realised moves and envelope breaks?),
   human agreement (overlap with my own top-5 on sampled days), catalyst recall
   (on scheduled-earnings days, does the post-mortem name the catalyst?),
   dispersion self-consistency (when it says "within dispersion," did the close
   actually land mid-distribution?), and base-rate validity (does a setup's
   claimed hit rate predict later, unseen outcomes?). Plus the **memory
   ablation**: run the replay with memory on and off and compare. If off wins,
   that is a finding.

## How good is the forecaster? (the honest baseline)

Before any of the agent machinery, `src/baseline.py` grades the raw Kronos-small
forecast over the whole window — 2,560 forecast-days, 40 names, **~11 months out
of sample** (Kronos's pretraining ends ~June 2024; arXiv:2508.02739).

| | result | read |
|---|---|---|
| median bias | mean signed error ≈ **0.0004** | unbiased |
| direction | median-sign hit **49.5%** vs 54.6% "always up" | **no directional edge** |
| calibration | 90% interval covers **78%**; PIT histogram U-shaped | **overconfident** |
| skill | Spearman(\|strength\|, \|actual ret\|) = **0.04** | strength barely orders eventful days |
| | Spearman(dispersion, \|actual ret\|) = **0.25** | the model *does* know when it's uncertain |

So Kronos on this task is an unbiased, low-skill forecaster with intervals ~12
points too tight. That is not a problem for the project — it is the premise. The
desk's job is disciplined review and accumulating memory around a mediocre model,
not alpha. Every eval metric is measured against that bar.

## What carried over, and the one trick

The whole deterministic spine survived the pivot from the project's v1: the
no-lookahead price loader, the hash-chained store, and the one non-obvious piece
of Kronos plumbing —

> `KronosPredictor.predict(sample_count=N)` runs N sampled paths and then
> averages them before returning. The vanilla call gives you a smoothed mean line
> and zero dispersion.

`sample_paths()` is the identical batched inference returning the array *before*
the mean. Thirty lines we own. That is the entire envelope, and the envelope is
the entire point.

## Results

*(to be filled from the completed replay: coverage, eventfulness table, catalyst
recall with Wilson intervals, the ablation.)*
