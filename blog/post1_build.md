# Narrative Arbitrage, Part 1: building a price-blind vs. news-blind disagreement engine

*Draft — live tracking starts the day this publishes. Part 2 is the results follow-up.*

---

## 1. The claim, and why daily

Every morning before the open I generate two independent forecasts for the same
set of stocks:

- **Structure** knows the price history and nothing else. It's
  [Kronos](https://github.com/shiyu-coder/Kronos), an open foundation model over
  tokenised candlesticks, run as a *distribution* — 200 sampled price paths, not
  a point estimate.
- **Narrative** knows the overnight headlines and nothing else — not the price,
  not the chart, not the recent return.

I log where they disagree, and at the close I check which one was right.

That's the whole project. **No trades, no position sizing, no P&L, no claim of
tradable alpha.** It's a pilot: about 20 trading days across 12 large-cap names,
roughly 240 signal-days, logged immutably and evaluated with honest confidence
intervals. Daily frequency is a deliberate choice — Kronos's context window is
512 bars, which at daily resolution is ~2 years of history, and an
overnight-headline → next-close horizon maps cleanly onto exactly one daily bar.

Why bother, if there's no trading? Because the *disagreement* is the interesting
object. A price-only model is structurally blind to a catalyst that hasn't
printed yet. A news-only reader is blind to the tape. When they diverge, one of
them is missing something — and which one, how often, and in what direction is a
measurable question.

## 2. Data, and the one rule that matters

The rule: **no lookahead, ever**. When I generate a signal *for* date `asof`,
Kronos may see bars strictly *before* `asof`. The first bar it predicts is `asof`
itself.

`src/data.py` enforces this at the slice and asserts it before returning:

```python
ctx = full[full.index < pd.Timestamp(asof)].tail(LOOKBACK_BARS)
if ctx.index.max() >= pd.Timestamp(asof):
    raise DataError("lookahead")
```

Prices come from Yahoo's chart API (hit through `curl_cffi` because Yahoo now
rate-limits the plain-Python user agent), adjusted for splits and dividends,
with a Stooq fallback and an inspectable CSV cache. Which source answered is
recorded on every row — a data-quality section in Part 2, not a footnote.

[FIGURE: the data summary table — 12 tickers, bar counts, date ranges, source]

## 3. Kronos, and the averaging trap

Kronos is decoder-only over a two-level K-line tokenizer. You give it history,
it autoregressively samples future tokens, the tokenizer decodes them back to
OHLCV. Crucially it's *probabilistic*: `predict(sample_count=N)` runs N sampled
paths.

Here's the trap. The upstream inference loop ends like this:

```python
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)   # <-- averages all N paths
```

The vanilla call **averages the samples before returning**. You get one smoothed
mean path and zero dispersion — no envelope, no quantiles, nothing to calibrate.
For a project whose entire premise is "where does the actual move fall in the
forecast distribution," that's fatal.

The fix is small: `src/kronos_infer.py::_auto_regressive_paths` is the identical
batched inference, returning the array *before* the mean — shape
`(sample_count, pred_len, 6)`. One forward pass, full distribution, ~30 lines I
own and can point at.

[FIGURE: 200 raw sampled close paths for NVDA, 2025-05-29]

**A second thing that would have quietly poisoned the results: lookback length.**
The obvious move is to feed Kronos its full 512-bar context — ~2 years of daily
history. That destabilises the sampler for any name trading near the high or low
of the window: the normalised last price reads as a multi-sigma extreme and the
model reverts hard, producing daily-return medians of −3% to −8% and tail paths
past −30%. Names sitting mid-range were fine; names in a sustained trend were
broken. Cutting the context to **120 bars (~6 months)** keeps the normalisation
local and the sampler well-behaved across all 12 tickers. That's the operating
point; the finding itself goes in Part 2's calibration section.

## 4. From paths to a distribution

With the paths in hand, the graded-bar close-to-close return becomes a
200-sample distribution. For **NVDA into its 2025-05-29 grade day**:

| quantile | return |
|---|---|
| 5% | −0.97% |
| 25% | +0.62% |
| 50% | +1.79% |
| 75% | +3.34% |
| 95% | +5.31% |

Median +1.79%, P(up) 85%, dispersion 1.9%. The "strength" of the structural
signal is median ÷ dispersion = +0.93 standard deviations.

[FIGURE: fan chart — context + sample paths + 5/25/50/75/95 bands]

Calibration check comes in Part 2: if the envelope is honest, the actual close
should land inside the 5–95 band about 90% of the time. (Spoiler from the three
case studies below: 2 of 3 — the small-n caveat is not rhetorical.)

## 5. The narrative side, kept genuinely blind

The narrative scorer sees a list of headline titles and summaries, the ticker,
and the date. Nothing else. It returns a structured object via an Anthropic tool
call — `direction ∈ {bear, neutral, bull}`, `conviction 1–10`, a rationale
naming the specific headlines, and the key drivers. The signal is
`direction × conviction`.

The system prompt is explicit that price talk is off-limits, and the exact
headline strings the model saw are logged with the score.

**An honesty problem I'm not going to paper over.** For a *historical* test date,
the model may already know how the stock moved — it's in the training data, and
there's no reliable way to detect or remove that. Scrubbing "shares rose 6%" out
of a headline doesn't fix it. So I don't pretend the case studies below are
clean. Every signal row carries a `llm_contaminated` flag (true when the date
precedes the model's knowledge cutoff). **The case studies are illustration. The
live run — every date after the cutoff — is the actual test.**

For NVDA 2025-05-29 the scorer sees three headlines: the top- and bottom-line
beat with data-center revenue +73% YoY; Q2 guidance of ~$45B *excluding* ~$8B of
lost H20 China revenue; and a $4.5B inventory charge tied to the April export
ban. Expected read: bullish, high conviction, tempered but not reversed by the
China caveat.

[FIGURE: the narrative agent's raw output for NVDA]

## 6. Divergence: definition and a worked verdict

Two signals, two ways to disagree:

- **Directional** — they disagree on sign: `sign(narrative) ≠ sign(median
  return)`, with narrative conviction ≥ 4.
- **Magnitude** — narrative is loud (conviction ≥ 7) but the move it implies
  sits in a low-probability region of Kronos's distribution:
  `P(return in narrative's direction) < 30%`.

The classification is deterministic and grounded entirely in the two signals'
numbers. Only the one-paragraph PM brief is an LLM call, and it's constrained to
restate and reconcile those numbers — not to add a third opinion.

Three worked days, none of which produced a divergence — which is itself worth
noting. On the "obvious" days the two views mostly agree; disagreement is rarer
than the framing suggests, and the month-long run is where it shows up.

**NVDA, 2025-05-29 — catalyst, both right.** Structure bullish (+1.79% median,
P(up) 85%); narrative bull, conviction 7 on the Q1 beat. Realised **+3.25%**,
inside the envelope (~90th percentile), both directions matched. A clean aligned
win — though Kronos was bullish only because NVDA was recovering off the April
lows, not because it knew about the earnings.

**NFLX, 2024-04-19 — narrative head-fake, both wrong.** The headlines were
unambiguous: 9.3M subscriber adds versus ~5M expected, EPS beat, margins up. The
scorer returned bull, conviction 7 — and its own rationale flagged the
buried "we'll stop reporting subscriber counts from 2025" line as an offset
before concluding the beat dominated. Structure, reading an uptrend, also leaned
bull (+1.13%). **The stock closed −9.1%**, straight through the 5th percentile of
Kronos's envelope. A double miss: −1 for both direction-hit rates and a coverage
failure on the structural side. This is what a news head-fake costs.

**JPM, 2025-06-11 — no catalyst.** The headline sub-agent dropped the generic
market recap, kept a routine analyst note, and the scorer judged it immaterial:
neutral, conviction 1. Narrative correctly abstains. Kronos leaned mildly up
(+0.33%); the stock finished −0.17%, near the centre of a tight envelope. On
quiet days the row is really just a calibration check, and this one passed.

## 7. Evaluation

At the close (technically the next morning, once the official close settles) the
second graph grades every locked signal:

- **envelope coverage** — fraction of actual closes inside the 5–95 band (target
  ~90%), per-ticker and pooled;
- **direction hit rate** — did each side call the sign? baseline 50%;
- **divergence-conditioned win rate** — on the rows where the two disagreed,
  which side matched the realised sign, with Wilson confidence intervals.

That last number is the headline research question, and with ~240 signal-days —
heavily autocorrelated across 12 names in one market regime — the CIs will be
wide. Part 2 will say so out loud.

## 8. Architecture

The nightly job is a [LangGraph](https://github.com/langchain-ai/langgraph) state
machine, not a linear script:

```
load_prices ─(ok)─> run_kronos ─> run_headlines ─> score_narrative
     │                                                   │
  (data error)                                            v
     └──────────────> handle_error <──────── assess_divergence
                          │                              │
                         END                          persist ─> END
```

The guiding principle: **an agent only where there's genuine reasoning or
tool-selection; everything else stays deterministic and is labelled as such.**
`run_headlines` is a real sub-agent (source selection branches on
historical-vs-live, dedupe, an LLM materiality filter, a clean no-news path). The
divergence brief is an LLM call. Price loading, Kronos inference, the divergence
*classification*, and persistence are plain functions. The restraint is
deliberate.

The signal log is append-only JSONL with a SHA-256 hash chained off each
previous row, committed to git after every run — so any later edit to history
breaks the chain and shows up in the diff.

The same two capabilities — `get_kronos_distribution` and `get_headlines` — are
exposed over **MCP**, so any MCP client can query them directly. A PM's
assistant could ask "what's Kronos's distribution on AVGO today, and what's the
overnight news" without touching this code.

## 9. Live tracking begins

The pipeline runs unattended for about a month on GitHub Actions. Part 2 reports:
envelope calibration, direction hit rates for both sides, the
divergence-conditioned win rate with its (wide) confidence interval, and an
honest accounting of how many scheduled runs actually fired.

The walkthrough notebook (`notebooks/walkthrough.ipynb`) runs the whole thing for
one ticker end to end if you want to see every intermediate object.
