"""Kronos structural forecaster — with the dispersion put back.

Why this file exists
--------------------
`KronosPredictor.predict(..., sample_count=N)` runs N sampled autoregressive
paths in one batched pass and then **averages them** before returning
(`np.mean(preds, axis=1)` at the end of `auto_regressive_inference`). The vanilla
call therefore yields a single smoothed mean path and *zero* dispersion — no
5th-95th envelope, which is the whole point of this project.

`sample_paths()` below runs the identical batched inference but returns the array
*before* the mean, shape `(sample_count, pred_len, 6)`. One pass, full
distribution, ~30 lines we own and can point at in the blog post.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import (
    GRADE_BAR,
    KRONOS_DEVICE,
    KRONOS_MODEL,
    KRONOS_SRC,
    KRONOS_TOKENIZER,
    MAX_CONTEXT,
    NB_OUT,
    PRED_LEN,
    QUANTILES,
    SAMPLE_COUNT,
    T,
    TOP_K,
    TOP_P,
)
from .data import PriceLoad, load_context

sys.path.insert(0, str(KRONOS_SRC))
from model.kronos import (  # noqa: E402
    KronosPredictor,
    calc_time_stamps,
    sample_from_logits,
)

PRICE_COLS = ["open", "high", "low", "close"]
FEAT_COLS = PRICE_COLS + ["volume", "amount"]


@lru_cache(maxsize=1)
def load_predictor() -> KronosPredictor:
    import torch
    from model.kronos import Kronos, KronosTokenizer

    tok = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER).eval()
    mdl = Kronos.from_pretrained(KRONOS_MODEL).eval()
    device = KRONOS_DEVICE
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    # .eval() matters on MPS: SDPA there rejects a non-zero dropout_p, and the
    # attention layer only zeroes dropout when not self.training.
    return KronosPredictor(mdl, tok, device=device, max_context=MAX_CONTEXT)


# --------------------------------------------------------------------------- #
# the vendored sampler: auto_regressive_inference minus the mean
# --------------------------------------------------------------------------- #
def _auto_regressive_paths(predictor, x, x_stamp, y_stamp, pred_len, *,
                           T=1.0, top_k=0, top_p=0.9, sample_count=200):
    """Byte-for-byte the upstream batched AR loop, except we keep every path.

    Returns ndarray of shape (batch, sample_count, pred_len, n_features).
    """
    import torch

    tokenizer, model = predictor.tokenizer, predictor.model
    max_context, clip = predictor.max_context, predictor.clip
    device = predictor.device

    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    x_stamp = torch.from_numpy(np.asarray(x_stamp, dtype=np.float32)).to(device)
    y_stamp = torch.from_numpy(np.asarray(y_stamp, dtype=np.float32)).to(device)

    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        B = x.size(0)
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(B, sample_count, z.size(1), z.size(2))  # <- keep the sample axis
        preds = z.detach().cpu().numpy()

    return preds[:, :, -pred_len:, :]  # (B, sample_count, pred_len, feat)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
@dataclass
class KronosForecast:
    ticker: str
    asof: date
    prev_close: float                    # last real close Kronos saw
    paths: np.ndarray                    # (sample_count, pred_len, 6) de-normalised prices
    y_index: pd.DatetimeIndex            # dates of the predicted bars
    context: pd.DataFrame                # what the model was fed (for plotting)

    # --- grade-bar close-return distribution -------------------------------
    @property
    def ret_dist(self) -> np.ndarray:
        """close-to-close return of the graded bar, one value per sampled path."""
        close = self.paths[:, GRADE_BAR - 1, PRICE_COLS.index("close")]
        return close / self.prev_close - 1.0

    @property
    def median_ret(self) -> float:
        return float(np.median(self.ret_dist))

    @property
    def std_ret(self) -> float:
        return float(np.std(self.ret_dist))

    @property
    def p_up(self) -> float:
        return float(np.mean(self.ret_dist > 0))

    @property
    def strength(self) -> float:
        """median move in units of the forecast's own dispersion (PLAN §6)."""
        return self.median_ret / self.std_ret if self.std_ret else 0.0

    def ret_quantiles(self, qs=QUANTILES) -> dict[float, float]:
        return {q: float(np.quantile(self.ret_dist, q)) for q in qs}

    def price_quantiles(self, qs=QUANTILES) -> pd.DataFrame:
        """per-step close-price quantiles, index = predicted dates."""
        close = self.paths[:, :, PRICE_COLS.index("close")]  # (samples, pred_len)
        return pd.DataFrame(
            {f"q{int(q * 100):02d}": np.quantile(close, q, axis=0) for q in qs},
            index=self.y_index,
        )

    def prob_direction(self, direction: int) -> float:
        """P(return has the given sign) under Kronos. direction in {-1, +1}."""
        if direction > 0:
            return float(np.mean(self.ret_dist > 0))
        if direction < 0:
            return float(np.mean(self.ret_dist < 0))
        return float(np.mean(np.abs(self.ret_dist) < self.std_ret))


def _next_bdays(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


def forecast(
    pl: PriceLoad,
    *,
    sample_count: int = SAMPLE_COUNT,
    pred_len: int = PRED_LEN,
    seed: int | None = None,
) -> KronosForecast:
    """Run Kronos on one ticker/asof and return the full path distribution."""
    import torch

    if seed is None:
        import hashlib
        h = hashlib.sha256(f"{pl.ticker}|{pl.asof.isoformat()}".encode()).hexdigest()
        seed = int(h[:8], 16)  # stable across processes (unlike builtin hash())
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31))

    predictor = load_predictor()
    df = pl.df[FEAT_COLS].astype(np.float32)
    x_index = pd.DatetimeIndex(df.index)
    y_index = _next_bdays(x_index[-1], pred_len)

    x = df.values
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    x_stamp = calc_time_stamps(pd.Series(x_index)).values.astype(np.float32)
    y_stamp = calc_time_stamps(pd.Series(y_index)).values.astype(np.float32)

    raw = _auto_regressive_paths(
        predictor,
        x_norm[None], x_stamp[None], y_stamp[None],
        pred_len, T=T, top_k=TOP_K, top_p=TOP_P, sample_count=sample_count,
    )[0]  # (sample_count, pred_len, feat)

    paths = raw * (x_std + 1e-5) + x_mean
    return KronosForecast(
        ticker=pl.ticker,
        asof=pl.asof,
        prev_close=float(df["close"].iloc[-1]),
        paths=paths,
        y_index=y_index,
        context=pl.df,
    )


# --------------------------------------------------------------------------- #
# fan chart
# --------------------------------------------------------------------------- #
def fan_chart(fc: KronosForecast, actual_close: float | None = None, *,
              context_bars: int = 60, path=None):
    import matplotlib.pyplot as plt

    ctx = fc.context["close"].iloc[-context_bars:]
    pq = fc.price_quantiles()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ctx.index, ctx.values, color="#222", lw=1.2, label="actual close (context)")

    # thin sample paths
    close_paths = fc.paths[:, :, PRICE_COLS.index("close")]
    for row in close_paths[:80]:
        ax.plot(fc.y_index, row, color="#4c78a8", alpha=0.06, lw=0.8)

    ax.fill_between(fc.y_index, pq["q05"], pq["q95"], color="#4c78a8", alpha=0.18, label="5-95%")
    ax.fill_between(fc.y_index, pq["q25"], pq["q75"], color="#4c78a8", alpha=0.30, label="25-75%")
    ax.plot(fc.y_index, pq["q50"], color="#4c78a8", lw=1.6, label="median")

    if actual_close is not None:
        ax.scatter([fc.y_index[GRADE_BAR - 1]], [actual_close], color="#e45756",
                   zorder=5, s=45, label=f"actual close ({actual_close:.2f})")

    ax.set_title(f"{fc.ticker} — Kronos forecast as of {fc.asof}  "
                 f"(median {fc.median_ret * 100:+.2f}%, P(up)={fc.p_up:.0%})")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig, ax


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run Kronos with full dispersion.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    ap.add_argument("--pred_len", type=int, default=PRED_LEN)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args(argv)

    pl = load_context(args.ticker, args.asof)
    fc = forecast(pl, sample_count=args.samples, pred_len=args.pred_len)

    print(f"{fc.ticker}  asof {fc.asof}  (context through {pl.last_context_date.date()}, "
          f"src={pl.source})")
    print(f"  prev_close      {fc.prev_close:.2f}")
    print(f"  grade bar       {fc.y_index[GRADE_BAR - 1].date()}")
    print(f"  median return   {fc.median_ret * 100:+.3f}%")
    print(f"  std / strength  {fc.std_ret * 100:.3f}%  /  {fc.strength:+.2f}")
    print(f"  P(up)           {fc.p_up:.1%}")
    print("  return quantiles:")
    for q, v in fc.ret_quantiles().items():
        print(f"    {int(q*100):>3}%   {v*100:+.3f}%")

    if args.chart:
        out = NB_OUT / f"fan_{fc.ticker}_{fc.asof}.png"
        fan_chart(fc, path=out)
        print(f"  chart -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
