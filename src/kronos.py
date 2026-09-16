"""Kronos structural forecaster — with the dispersion put back.

`KronosPredictor.predict(..., sample_count=N)` runs N sampled autoregressive
paths in one batched pass and then **averages them** before returning
(`np.mean(preds, axis=1)`). The vanilla call yields a single smoothed mean path
and *zero* dispersion.

`sample_paths()` runs the identical batched inference and returns the array
*before* the mean, shape `(sample_count, pred_len, 6)` — the full predictive
distribution. This is the only Kronos code we own; everything else is vendored
in `vendor_kronos/` (cloned by `scripts/setup.sh`, not part of this repo).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))   # weights stay in-project
KRONOS_SRC = ROOT / "vendor_kronos"

KRONOS_MODEL = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE")            # None -> auto (mps / cpu)
MAX_CONTEXT = 512                                     # Kronos-small / base window
SAMPLE_COUNT, T, TOP_K, TOP_P = 50, 1.0, 0, 0.9        # sample_paths() defaults

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

    x / x_stamp / y_stamp are (batch, seq, feat). Returns
    (batch, sample_count, pred_len, n_features).
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
        z = z.reshape(B, sample_count, z.size(1), z.size(2))   # keep the sample axis
        preds = z.detach().cpu().numpy()

    return preds[:, :, -pred_len:, :]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def sample_paths(
    context: pd.DataFrame,
    y_index: pd.DatetimeIndex,
    *,
    sample_count: int = SAMPLE_COUNT,
    predictor=None,
) -> np.ndarray:
    """Full predictive path distribution for one (name, origin).

    `context` — OHLCV(+amount) bars strictly BEFORE the forecast origin, tz-naive
                or tz-aware DatetimeIndex, most-recent last.
    `y_index` — the timestamps of the bars to forecast (length = pred_len).

    Returns `(sample_count, pred_len, 6)` of de-normalised OHLCV+amount prices.
    Per-window z-normalisation matches the upstream `KronosPredictor`.
    """
    predictor = predictor or load_predictor()
    df = context[FEAT_COLS].astype(np.float32)
    x = df.to_numpy()
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    ctx_ts = pd.DatetimeIndex(df.index)
    if ctx_ts.tz is not None:
        ctx_ts = ctx_ts.tz_localize(None)
    y_ts = pd.DatetimeIndex(y_index)
    if y_ts.tz is not None:
        y_ts = y_ts.tz_localize(None)

    x_stamp = calc_time_stamps(pd.Series(ctx_ts)).values.astype(np.float32)
    y_stamp = calc_time_stamps(pd.Series(y_ts)).values.astype(np.float32)

    raw = _auto_regressive_paths(
        predictor, x_norm[None], x_stamp[None], y_stamp[None], len(y_ts),
        T=T, top_k=TOP_K, top_p=TOP_P, sample_count=sample_count,
    )[0]
    return raw * (x_std + 1e-5) + x_mean
