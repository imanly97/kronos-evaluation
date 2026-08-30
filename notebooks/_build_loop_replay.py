"""Generate notebooks/loop_replay.ipynb — the window replayed, memory accumulating,
and the memory on/off ablation (PLAN §7).

    .venv/bin/python notebooks/_build_loop_replay.py

Assumes the forecast cache for the replay window is already built
(`python -m src.forecasts`). The replay itself is long (agent calls per session) —
the notebook reads the *logs* a completed replay produced rather than running it
inline; a cell shows the command.
"""
from __future__ import annotations

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md("""# Forecast Desk — loop replay & the memory ablation

The loop replayed day by day over the locked window. Desk memory accumulates
exactly as it would live: each post-mortem writes episodes, the next morning's
triage reads the derived stats back.

**Run the replay first** (long — LLM calls per session):

```bash
python -m src.forecasts                       # 1. deterministic cache (once)
python -m src.run_daily replay --skip-forecast # 2. the loop, memory ON
```

For the ablation, a second pass with memory OFF into a scratch log dir that
reuses the same forecast cache:

```bash
mkdir -p _ablation/log _ablation/mem && cp log/forecasts.jsonl _ablation/log/
FMD_LOG_DIR=$PWD/_ablation/log FMD_MEMORY_DIR=$PWD/_ablation/mem \\
  python -m src.run_daily replay --skip-forecast --no-memory
```
""")

    code("""import os
if "src" not in os.listdir("."):
    os.chdir("..")""")
    code("""import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src import config
from src.store import load, joined""")

    md("## 1 · Coverage & the running grade")
    code("""from src.grade import report
print(report())
j = joined(); j = j[j["ret"].notna()]
print("\\nsessions:", j["asof"].nunique(), "| graded forecast-days:", len(j))""")

    md("## 2 · Memory accumulating")
    code("""eps = [json.loads(l) for l in (config.MEMORY_DIR / "episodes.jsonl").read_text().splitlines() if l.strip()]
e = pd.DataFrame(eps)
e["asof"] = pd.to_datetime(e["asof"])
by_day = e.groupby("asof").size().cumsum()
by_day.plot(title="cumulative desk-memory episodes"); plt.ylabel("episodes"); plt.show()
e["classification"].value_counts()""")

    md("## 3 · Eventfulness — did flagged names skew eventful?")
    code("""from eval.evaluate import triage_eventfulness
pd.Series(triage_eventfulness())""")

    md("## 4 · The ablation — memory ON vs OFF")
    code("""abl = Path("_ablation/log/watchlist.jsonl")
if not abl.exists():
    print("run the --no-memory pass first (see the command at the top)")
else:
    on  = load("watchlist")
    off = pd.read_json(abl, lines=True, dtype={"asof": str})
    on_set  = set(zip(on["asof"].astype(str),  on["ticker"]))
    off_set = set(zip(off["asof"].astype(str), off["ticker"]))
    print(f"memory ON  picks: {len(on_set)}")
    print(f"memory OFF picks: {len(off_set)}")
    print(f"overlap:          {len(on_set & off_set)}")
    print(f"only with memory: {sorted(t for _,t in (on_set - off_set))[:20]}")
    print(f"only without:     {sorted(t for _,t in (off_set - on_set))[:20]}")""")

    code("""# eventfulness of each arm's flagged set (needs both replays graded)
from eval.evaluate import triage_eventfulness
import os
print("memory ON :", triage_eventfulness())
os.environ["FMD_LOG_DIR"] = str(Path("_ablation/log").resolve())
import importlib, src.config, src.store, eval.evaluate as ev
for m in (src.config, src.store, ev): importlib.reload(m)
print("memory OFF:", ev.triage_eventfulness())""")

    md("""---
**Reading the ablation:** if the memory-OFF arm's flagged names are just as
eventful and just as close to your own picks, that is a reportable finding —
memory added process, not signal (PLAN §14). If memory-ON skews more eventful or
tracks your picks better, the feedback edge is doing work.""")
    return nb


def main() -> None:
    nb = build()
    out = __import__("pathlib").Path(__file__).parent / "loop_replay.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
