"""
Pure liquidity-grab / liquidity-sweep strategy logic.

This module is intentionally exchange-agnostic: it consumes OHLCV pandas DataFrames
and returns structured signal dicts for the rest of the bot to execute/backtest.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    # config/ has no __init__.py, but Python namespace packages still allow this import in most cases.
    from config.settings import (
        ATR_PERIOD as DEFAULT_ATR_PERIOD,
        MIN_RR as DEFAULT_MIN_RR,
        SWEEP_BUFFER_PCT as DEFAULT_SWEEP_BUFFER_PCT,
        SWING_LOOKBACK as DEFAULT_SWING_LOOKBACK,
    )
except Exception:
    # Safe fallbacks so the strategy can run in isolation.
    DEFAULT_ATR_PERIOD = 14
    DEFAULT_MIN_RR = 2.0
    DEFAULT_SWEEP_BUFFER_PCT = 0.001
    DEFAULT_SWING_LOOKBACK = 20


_REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close")


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _compute_atr(df: pd.DataFrame, atr_period: int) -> pd.Series:
    _require_columns(df, ("high", "low", "close"))
    atr = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_period)
    # pandas_ta commonly names it ATR_{length}, but be defensive.
    if isinstance(atr, pd.Series):
        return atr.rename("atr")
    if isinstance(atr, pd.DataFrame):
        if atr.shape[1] != 1:
            raise ValueError("Unexpected ATR output shape from pandas_ta.")
        return atr.iloc[:, 0].rename("atr")
    raise ValueError("Unexpected ATR output type from pandas_ta.")


def detect_swings(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    """
    Detect swing highs/lows using a centered rolling-window fractal.

    Note: swing identification uses future bars (because it's centered). When generating
    signals, we therefore gate by `swing_right` and `displacement_bars` so no lookahead
    is used.
    """
    _require_columns(df, ("high", "low"))
    out = df.copy()
    window = left + right + 1
    out["swing_high"] = out["high"] == out["high"].rolling(window=window, center=True).max()
    out["swing_low"] = out["low"] == out["low"].rolling(window=window, center=True).min()
    return out


def _compute_respected_swings(
    df: pd.DataFrame,
    *,
    left: int,
    right: int,
    atr_period: int,
    displacement_atr_mult: float,
    displacement_bars: int,
    require_close_move: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Respect filter (playbook rule):
    - A swing is "respected" only if price moved away at least `displacement_atr_mult * ATR`
      after touching it.
    """
    _require_columns(df, _REQUIRED_OHLCV_COLUMNS)
    atr = _compute_atr(df, atr_period)
    swings = detect_swings(df, left=left, right=right)

    respected_low = pd.Series(False, index=df.index)
    respected_high = pd.Series(False, index=df.index)

    # Iterate only over swing candidates (usually far fewer than all bars).
    swing_low_positions = np.flatnonzero(swings["swing_low"].to_numpy())
    swing_high_positions = np.flatnonzero(swings["swing_high"].to_numpy())

    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()

    for pos in swing_low_positions:
        atr_s = atr.iloc[pos]
        if pd.isna(atr_s) or atr_s <= 0:
            continue
        # Need future bars to confirm "move away".
        end = pos + 1 + displacement_bars
        if end > len(df):
            continue

        touched = lows[pos]
        max_high_after = float(np.max(highs[pos + 1 : end]))
        if (max_high_after - touched) < displacement_atr_mult * float(atr_s):
            continue
        if require_close_move:
            max_close_after = float(np.max(closes[pos + 1 : end]))
            if (max_close_after - touched) < displacement_atr_mult * float(atr_s):
                continue
        respected_low.iloc[pos] = True

    for pos in swing_high_positions:
        atr_s = atr.iloc[pos]
        if pd.isna(atr_s) or atr_s <= 0:
            continue
        end = pos + 1 + displacement_bars
        if end > len(df):
            continue

        touched = highs[pos]
        min_low_after = float(np.min(lows[pos + 1 : end]))
        if (touched - min_low_after) < displacement_atr_mult * float(atr_s):
            continue
        if require_close_move:
            min_close_after = float(np.min(closes[pos + 1 : end]))
            if (touched - min_close_after) < displacement_atr_mult * float(atr_s):
                continue
        respected_high.iloc[pos] = True

    return respected_low, respected_high


def _body_ratio(df: pd.DataFrame, i: int) -> float:
    """
    Candle body ratio: body / range.
    Used as the "strong reversal" confirmation (>= 0.60 by default).
    """
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    h = float(df["high"].iloc[i])
    l = float(df["low"].iloc[i])
    rng = max(h - l, 1e-12)
    return abs(c - o) / rng


def _calculate_signal_metrics(
    *,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    min_rr: float,
) -> dict[str, Any]:
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be 'LONG' or 'SHORT'")

    if direction == "LONG":
        stop_dist = float(entry_price - stop_loss)
        target_dist = float(target_price - entry_price)
    else:
        stop_dist = float(stop_loss - entry_price)
        target_dist = float(entry_price - target_price)

    if stop_dist <= 0 or target_dist <= 0:
        return {"valid": False, "rr_ratio": None, "stop_distance": stop_dist, "target_distance": target_dist}

    rr = target_dist / stop_dist
    if rr < min_rr:
        return {"valid": False, "rr_ratio": rr, "stop_distance": stop_dist, "target_distance": target_dist}

    return {"valid": True, "rr_ratio": rr, "stop_distance": stop_dist, "target_distance": target_dist}


def _last_position_at_or_before(positions: np.ndarray, values: np.ndarray, threshold_pos: int) -> Optional[tuple[int, float]]:
    if positions.size == 0:
        return None
    idx = bisect_right(positions.tolist(), threshold_pos) - 1
    if idx < 0:
        return None
    return int(positions[idx]), float(values[idx])


def _find_nearest_opposing_swing(
    *,
    direction: str,
    threshold_pos: int,
    entry_price: float,
    atr_i: float,
    swing_positions: np.ndarray,
    swing_levels: np.ndarray,
    target_max_distance_atr_mult: float,
) -> Optional[tuple[int, float]]:
    """
    Target rule (your requirement):
    - LONG: nearest previous respected swing HIGH that is above entry_price.
    - SHORT: nearest previous respected swing LOW that is below entry_price.
    """
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be 'LONG' or 'SHORT'")

    # Start from the nearest previous swing (pos <= threshold_pos) and walk backward until it fits constraints.
    idx = bisect_right(swing_positions.tolist(), threshold_pos) - 1
    if idx < 0:
        return None

    max_dist = float(target_max_distance_atr_mult) * float(atr_i)
    while idx >= 0:
        pos = int(swing_positions[idx])
        level = float(swing_levels[idx])
        if direction == "LONG":
            if level > entry_price:
                dist = level - entry_price
                if dist <= max_dist:
                    return pos, level
                # If even this nearest candidate is too far, earlier ones are likely also too far,
                # but not guaranteed; keep scanning a bit.
        else:
            if level < entry_price:
                dist = entry_price - level
                if dist <= max_dist:
                    return pos, level
        idx -= 1

    return None


def _compute_market_structure_bias(
    htf_df: pd.DataFrame,
    *,
    left: int,
    right: int,
    atr_period: int,
    displacement_atr_mult: float,
    displacement_bars: int,
) -> pd.Series:
    """
    HTF bias (your requirement):
    - Bullish if last two swing highs are higher highs and last two swing lows are higher lows.
    - Bearish if last two swing highs are lower highs and last two swing lows are lower lows.
    """
    _require_columns(htf_df, _REQUIRED_OHLCV_COLUMNS)
    respected_low, respected_high = _compute_respected_swings(
        htf_df,
        left=left,
        right=right,
        atr_period=atr_period,
        displacement_atr_mult=displacement_atr_mult,
        displacement_bars=displacement_bars,
        require_close_move=True,
    )
    bias = pd.Series("neutral", index=htf_df.index, dtype=object)

    atr_confirm_bars = max(right, displacement_bars)
    highs_positions = np.flatnonzero(respected_high.to_numpy())
    lows_positions = np.flatnonzero(respected_low.to_numpy())
    highs_values = htf_df["high"].to_numpy()[highs_positions]
    lows_values = htf_df["low"].to_numpy()[lows_positions]

    high_deque: deque[float] = deque(maxlen=2)
    low_deque: deque[float] = deque(maxlen=2)

    hi_ptr = 0
    lo_ptr = 0
    for cur_pos in range(len(htf_df)):
        threshold = cur_pos - atr_confirm_bars
        while hi_ptr < len(highs_positions) and int(highs_positions[hi_ptr]) <= threshold:
            high_deque.append(float(highs_values[hi_ptr]))
            hi_ptr += 1
        while lo_ptr < len(lows_positions) and int(lows_positions[lo_ptr]) <= threshold:
            low_deque.append(float(lows_values[lo_ptr]))
            lo_ptr += 1

        if len(high_deque) < 2 or len(low_deque) < 2:
            bias.iloc[cur_pos] = "neutral"
        else:
            hh0, hh1 = high_deque[0], high_deque[1]
            ll0, ll1 = low_deque[0], low_deque[1]
            if hh1 > hh0 and ll1 > ll0:
                bias.iloc[cur_pos] = "bullish"
            elif hh1 < hh0 and ll1 < ll0:
                bias.iloc[cur_pos] = "bearish"
            else:
                bias.iloc[cur_pos] = "neutral"

    return bias


def _map_htf_bias_to_base(
    base_df: pd.DataFrame, htf_df: pd.DataFrame, htf_bias: pd.Series
) -> pd.Series:
    """
    Map HTF bias series onto base timeframe rows.

    Requires both DataFrames to have DatetimeIndex for correct alignment.
    """
    if not isinstance(base_df.index, pd.DatetimeIndex) or not isinstance(htf_df.index, pd.DatetimeIndex):
        raise ValueError("htf_df and df must both have DatetimeIndex for HTF bias alignment.")

    base_times = base_df.index.view("i8")
    htf_times = htf_df.index.view("i8")

    # For each base time, take the most recent HTF time <= base time.
    positions = np.searchsorted(htf_times, base_times, side="right") - 1
    mapped = np.empty(len(base_df), dtype=object)
    mapped[:] = "neutral"
    valid = positions >= 0
    if np.any(valid):
        mapped[valid] = htf_bias.iloc[positions[valid]].to_numpy()
    return pd.Series(mapped, index=base_df.index, dtype=object)


def generate_signals(
    df: pd.DataFrame,
    htf_df: Optional[pd.DataFrame] = None,
    *,
    swing_left: Optional[int] = None,
    swing_right: Optional[int] = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    displacement_atr_mult: float = 0.8,
    displacement_bars: Optional[int] = None,
    body_ratio_threshold: float = 0.50,
    sweep_buffer_pct: float = 0.0005,
    stop_buffer_pct: Optional[float] = None,
    entry_max_distance_atr_mult: float = 1.5,
    target_max_distance_atr_mult: float = 5.0,
    min_rr: float = 1.5,
) -> list[dict[str, Any]]:
    """
    Return liquidity-grab trade signals with RR >= `min_rr`.

    Signal dict keys:
    - direction: "LONG" | "SHORT"
    - entry_price, stop_loss, target_price, rr_ratio
    - swept_level, swing_level_swept_index
    - target_swing_level, target_swing_index
    - atr, htf_bias, candle_body_ratio, and other metadata
    """
    _require_columns(df, _REQUIRED_OHLCV_COLUMNS)
    if len(df) < 50:
        return []

    if swing_left is None or swing_right is None:
        # Map configured swing lookback into a workable left/right window.
        total_window = int(DEFAULT_SWING_LOOKBACK)
        left = max(2, total_window // 2)
        right = max(2, total_window // 2)
    else:
        left = int(swing_left)
        right = int(swing_right)

    if displacement_bars is None:
        displacement_bars = 5
    displacement_bars = int(displacement_bars)

    if stop_buffer_pct is None:
        stop_buffer_pct = sweep_buffer_pct

    atr = _compute_atr(df, atr_period)
    respected_low, respected_high = _compute_respected_swings(
        df,
        left=left,
        right=right,
        atr_period=atr_period,
        displacement_atr_mult=displacement_atr_mult,
        displacement_bars=displacement_bars,
        # Use wick-based displacement to avoid missing setups due to strict close-based rules.
        require_close_move=False,
    )

    low_positions = np.flatnonzero(respected_low.to_numpy())
    low_values = df["low"].to_numpy()[low_positions]
    high_positions = np.flatnonzero(respected_high.to_numpy())
    high_values = df["high"].to_numpy()[high_positions]

    # HTF bias gating (structure-first).
    if htf_df is not None:
        htf_bias = _compute_market_structure_bias(
            htf_df,
            left=left,
            right=right,
            atr_period=atr_period,
            displacement_atr_mult=displacement_atr_mult,
            displacement_bars=displacement_bars,
        )
        base_bias = _map_htf_bias_to_base(df, htf_df, htf_bias)
    else:
        base_bias = pd.Series("neutral", index=df.index, dtype=object)

    swing_confirm_bars = max(right, displacement_bars)
    start_pos = max(swing_confirm_bars + 1, atr_period + 1)

    signals: list[dict[str, Any]] = []
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()

    for i in range(start_pos, len(df)):
        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        confirm_threshold = i - swing_confirm_bars
        if confirm_threshold < 0:
            continue

        candle_body_ratio = float(_body_ratio(df, i))
        o = float(opens[i])
        c = float(closes[i])
        h = float(highs[i])
        l = float(lows[i])
        if h - l <= 0:
            continue

        bias = str(base_bias.iloc[i])
        close_to_sweep_ok_for_long = True
        close_to_sweep_ok_for_short = True

        # LONG setup: sweep below last respected swing low + bullish strong candle.
        # Allow HTF neutral for initial calibration.
        if bias in {"bullish", "neutral"}:
            last_low = _last_position_at_or_before(low_positions, low_values, confirm_threshold)
            if last_low is not None:
                swept_pos, swept_level = last_low
                swept = l < swept_level * (1.0 - sweep_buffer_pct)
                reversal = (c > o) and (candle_body_ratio > body_ratio_threshold)
                if swept and reversal:
                    entry_price = c  # enter on close of confirmation candle
                    # Enter "near" the swept level (avoid chasing).
                    if (entry_price - swept_level) > entry_max_distance_atr_mult * atr_i:
                        close_to_sweep_ok_for_long = False
                    # Stop should be beyond the sweep candle extreme (tighter but safe).
                    stop_loss = l * (1.0 - stop_buffer_pct)

                    target = _find_nearest_opposing_swing(
                        direction="LONG",
                        threshold_pos=confirm_threshold,
                        entry_price=entry_price,
                        atr_i=atr_i,
                        swing_positions=high_positions,
                        swing_levels=high_values,
                        target_max_distance_atr_mult=target_max_distance_atr_mult,
                    )
                    if close_to_sweep_ok_for_long and target is not None:
                        target_pos, target_level = target
                        metrics = _calculate_signal_metrics(
                            direction="LONG",
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target_price=target_level,
                            min_rr=min_rr,
                        )
                        if metrics["valid"]:
                            signals.append(
                                {
                                    "direction": "LONG",
                                    "signal_index": df.index[i],
                                    "entry_price": float(entry_price),
                                    "stop_loss": float(stop_loss),
                                    "target_price": float(target_level),
                                    "rr_ratio": float(metrics["rr_ratio"]),
                                    "atr": float(atr_i),
                                    "htf_bias": bias,
                                    "candle_body_ratio": float(candle_body_ratio),
                                    "swing_level_swept": float(swept_level),
                                    "swing_level_swept_index": df.index[swept_pos],
                                    "target_swing_level": float(target_level),
                                    "target_swing_index": df.index[target_pos],
                                    "sweep_buffer_pct": float(sweep_buffer_pct),
                                    "displacement_atr_mult": float(displacement_atr_mult),
                                }
                            )

        # SHORT setup: sweep above last respected swing high + bearish strong candle.
        # Allow HTF neutral for initial calibration.
        if bias in {"bearish", "neutral"}:
            last_high = _last_position_at_or_before(high_positions, high_values, confirm_threshold)
            if last_high is not None:
                swept_pos, swept_level = last_high
                swept = h > swept_level * (1.0 + sweep_buffer_pct)
                reversal = (c < o) and (candle_body_ratio > body_ratio_threshold)
                if swept and reversal:
                    entry_price = c
                    if (swept_level - entry_price) > entry_max_distance_atr_mult * atr_i:
                        close_to_sweep_ok_for_short = False
                    stop_loss = h * (1.0 + stop_buffer_pct)

                    target = _find_nearest_opposing_swing(
                        direction="SHORT",
                        threshold_pos=confirm_threshold,
                        entry_price=entry_price,
                        atr_i=atr_i,
                        swing_positions=low_positions,
                        swing_levels=low_values,
                        target_max_distance_atr_mult=target_max_distance_atr_mult,
                    )
                    if close_to_sweep_ok_for_short and target is not None:
                        target_pos, target_level = target
                        metrics = _calculate_signal_metrics(
                            direction="SHORT",
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target_price=target_level,
                            min_rr=min_rr,
                        )
                        if metrics["valid"]:
                            signals.append(
                                {
                                    "direction": "SHORT",
                                    "signal_index": df.index[i],
                                    "entry_price": float(entry_price),
                                    "stop_loss": float(stop_loss),
                                    "target_price": float(target_level),
                                    "rr_ratio": float(metrics["rr_ratio"]),
                                    "atr": float(atr_i),
                                    "htf_bias": bias,
                                    "candle_body_ratio": float(candle_body_ratio),
                                    "swing_level_swept": float(swept_level),
                                    "swing_level_swept_index": df.index[swept_pos],
                                    "target_swing_level": float(target_level),
                                    "target_swing_index": df.index[target_pos],
                                    "sweep_buffer_pct": float(sweep_buffer_pct),
                                    "displacement_atr_mult": float(displacement_atr_mult),
                                }
                            )

    return signals