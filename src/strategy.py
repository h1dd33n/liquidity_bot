"""
Pure liquidity-grab / liquidity-sweep strategy logic (fixed version).
No RR prediction inside strategy — only liquidity-based targets.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from config.settings import (
        ATR_PERIOD as DEFAULT_ATR_PERIOD,
        MIN_RR as DEFAULT_MIN_RR,
        SWEEP_BUFFER_PCT as DEFAULT_SWEEP_BUFFER_PCT,
        SWING_LOOKBACK as DEFAULT_SWING_LOOKBACK,
    )
except Exception:
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
    if isinstance(atr, pd.Series):
        return atr.rename("atr")
    if isinstance(atr, pd.DataFrame):
        return atr.iloc[:, 0].rename("atr")
    raise ValueError("Unexpected ATR output from pandas_ta.")


def detect_swings(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
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
    require_close_move: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Respected swing filter (playbook rule):
    - A swing is respected if price moved away by at least:
        displacement_atr_mult * ATR
      within the next `displacement_bars` after touching the swing.
    """
    _require_columns(df, _REQUIRED_OHLCV_COLUMNS)
    atr = _compute_atr(df, atr_period)
    swings = detect_swings(df, left=left, right=right)

    respected_low = pd.Series(False, index=df.index)
    respected_high = pd.Series(False, index=df.index)

    swing_low_positions = np.flatnonzero(swings["swing_low"].to_numpy())
    swing_high_positions = np.flatnonzero(swings["swing_high"].to_numpy())

    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()

    for pos in swing_low_positions:
        atr_s = float(atr.iloc[pos])
        if not np.isfinite(atr_s) or atr_s <= 0:
            continue

        end = pos + 1 + displacement_bars
        if end > len(df):
            continue

        touched = float(lows[pos])
        max_high_after = float(np.max(highs[pos + 1 : end]))
        if (max_high_after - touched) < float(displacement_atr_mult) * atr_s:
            continue
        if require_close_move:
            max_close_after = float(np.max(closes[pos + 1 : end]))
            if (max_close_after - touched) < float(displacement_atr_mult) * atr_s:
                continue
        respected_low.iloc[pos] = True

    for pos in swing_high_positions:
        atr_s = float(atr.iloc[pos])
        if not np.isfinite(atr_s) or atr_s <= 0:
            continue

        end = pos + 1 + displacement_bars
        if end > len(df):
            continue

        touched = float(highs[pos])
        min_low_after = float(np.min(lows[pos + 1 : end]))
        if (touched - min_low_after) < float(displacement_atr_mult) * atr_s:
            continue
        if require_close_move:
            min_close_after = float(np.min(closes[pos + 1 : end]))
            if (touched - min_close_after) < float(displacement_atr_mult) * atr_s:
                continue
        respected_high.iloc[pos] = True

    return respected_low, respected_high


def _body_ratio(df: pd.DataFrame, i: int) -> float:
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    h = float(df["high"].iloc[i])
    l = float(df["low"].iloc[i])
    rng = max(h - l, 1e-12)
    return abs(c - o) / rng


def _last_position_at_or_before(
    positions: np.ndarray,
    values: np.ndarray,
    threshold_pos: int,
) -> Optional[tuple[int, float]]:
    if positions.size == 0:
        return None
    idx = bisect_right(positions.tolist(), threshold_pos) - 1
    if idx < 0:
        return None
    return int(positions[idx]), float(values[idx])


def _find_nearest_opposing_swing(   # simplified — no RR check here
    *,
    direction: str,
    threshold_pos: int,
    entry_price: float,
    atr_i: float,
    swing_positions: np.ndarray,
    swing_levels: np.ndarray,
    target_max_distance_atr_mult: float = 8.0,
) -> Optional[tuple[int, float]]:
    """Return nearest previous respected opposing swing (liquidity pool)."""
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be 'LONG' or 'SHORT'")

    idx = bisect_right(swing_positions.tolist(), threshold_pos) - 1
    if idx < 0:
        return None

    max_dist = float(target_max_distance_atr_mult) * float(atr_i)

    while idx >= 0:
        pos = int(swing_positions[idx])
        level = float(swing_levels[idx])
        if direction == "LONG":
            if level > entry_price and (level - entry_price) <= max_dist:
                return pos, level
        else:  # SHORT
            if level < entry_price and (entry_price - level) <= max_dist:
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
    HTF bias (structure-first):
    - bullish: last two respected swing highs are higher highs AND last two
               respected swing lows are higher lows.
    - bearish: last two respected swing highs are lower highs AND last two
               respected swing lows are lower lows.
    """
    _require_columns(htf_df, _REQUIRED_OHLCV_COLUMNS)

    respected_low, respected_high = _compute_respected_swings(
        htf_df,
        left=left,
        right=right,
        atr_period=atr_period,
        displacement_atr_mult=displacement_atr_mult,
        displacement_bars=displacement_bars,
        require_close_move=False,
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
    base_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_bias: pd.Series,
) -> pd.Series:
    """
    Map HTF bias series onto base timeframe rows by taking the most recent
    HTF timestamp <= base timestamp.
    """
    if not isinstance(base_df.index, pd.DatetimeIndex) or not isinstance(htf_df.index, pd.DatetimeIndex):
        raise ValueError("htf_df and df must both have DatetimeIndex for HTF bias alignment.")

    base_times = base_df.index.view("i8")
    htf_times = htf_df.index.view("i8")

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
    displacement_atr_mult: float = 1.0,
    displacement_bars: Optional[int] = None,
    body_ratio_threshold: float = 0.55,
    sweep_buffer_pct: float = DEFAULT_SWEEP_BUFFER_PCT,
    stop_buffer_pct: Optional[float] = None,
    entry_max_distance_atr_mult: float = 0.8,
    target_max_distance_atr_mult: float = 8.0,
    min_rr: float = DEFAULT_MIN_RR,  # used only for rr_ratio metadata (risk_manager enforces actual filtering)
    allow_neutral_htf_in_backtest: bool = False,
    debug: bool = False,
) -> list[dict[str, Any]]:
    _require_columns(df, _REQUIRED_OHLCV_COLUMNS)
    if len(df) < 50:
        return []

    if swing_left is None or swing_right is None:
        total_window = int(DEFAULT_SWING_LOOKBACK)
        swing_left = max(2, total_window // 2)
        swing_right = max(2, total_window // 2)

    left = int(swing_left)
    right = int(swing_right)

    if displacement_bars is None:
        displacement_bars = 5
    displacement_bars = int(displacement_bars)

    if stop_buffer_pct is None:
        stop_buffer_pct = abs(float(sweep_buffer_pct))
        if stop_buffer_pct == 0:
            stop_buffer_pct = float(DEFAULT_SWEEP_BUFFER_PCT)
    stop_buffer_pct = float(stop_buffer_pct)

    atr = _compute_atr(df, atr_period)

    respected_low, respected_high = _compute_respected_swings(
        df,
        left=left,
        right=right,
        atr_period=atr_period,
        displacement_atr_mult=float(displacement_atr_mult),
        displacement_bars=int(displacement_bars),
        require_close_move=False,
    )

    low_positions = np.flatnonzero(respected_low.to_numpy())
    low_values = df["low"].to_numpy()[low_positions]
    high_positions = np.flatnonzero(respected_high.to_numpy())
    high_values = df["high"].to_numpy()[high_positions]

    # HTF bias (structure-first). For calibration, allow neutral to pass.
    if htf_df is not None:
        htf_bias = _compute_market_structure_bias(
            htf_df,
            left=left,
            right=right,
            atr_period=atr_period,
            displacement_atr_mult=float(displacement_atr_mult),
            displacement_bars=int(displacement_bars),
        )
        base_bias = _map_htf_bias_to_base(df, htf_df, htf_bias)
    else:
        base_bias = pd.Series("neutral", index=df.index, dtype=object)

    # We cannot safely reference swings earlier than the swing-right horizon.
    start_pos = max(int(atr_period) + int(right) + int(displacement_bars), 50)

    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    signals: list[dict[str, Any]] = []

    # Debug counters to understand why signals are (or aren't) produced.
    long_sweeps = 0
    long_reversals = 0
    long_sweep_reversals = 0
    long_target_search_attempts = 0
    long_targets_none = 0
    long_targets_found = 0
    long_chase_rejects = 0

    short_sweeps = 0
    short_reversals = 0
    short_sweep_reversals = 0
    short_target_search_attempts = 0
    short_targets_none = 0
    short_targets_found = 0
    short_chase_rejects = 0

    for i in range(start_pos, len(df)):
        pivot_threshold_pos = i - right
        if pivot_threshold_pos < 0:
            continue

        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        o = float(opens[i])
        c = float(closes[i])
        h = float(highs[i])
        l = float(lows[i])
        if h - l <= 0:
            continue

        candle_body_ratio = float(_body_ratio(df, i))
        bias = str(base_bias.iloc[i]) if htf_df is not None else "neutral"

        allow_long = False
        allow_short = False
        if bias == "bullish":
            allow_long = True
        elif bias == "bearish":
            allow_short = True
        else:
            if allow_neutral_htf_in_backtest:
                allow_long = True
                allow_short = True

        # LONG setup
        if allow_long:
            last_low = _last_position_at_or_before(low_positions, low_values, pivot_threshold_pos)
            if last_low is not None:
                swept_pos, swept_level = last_low
                swept = l < float(swept_level) * (1.0 - float(sweep_buffer_pct))
                # Reversal must reclaim the swept liquidity level,
                # otherwise the stop can end up almost on top of entry.
                reversal = (c > o) and (c > float(swept_level)) and (candle_body_ratio > float(body_ratio_threshold))
                if swept and reversal:
                    long_sweeps += 1
                    long_reversals += 1
                    long_sweep_reversals += 1

                    entry_price = float(c)
                    if (entry_price - float(swept_level)) > float(entry_max_distance_atr_mult) * atr_i:
                        long_chase_rejects += 1
                        # Too far from liquidity: do not take this setup.
                    else:
                        stop_loss = float(swept_level) * (1.0 - stop_buffer_pct)
                        if stop_loss < entry_price:
                            long_target_search_attempts += 1
                            target = _find_nearest_opposing_swing(
                                direction="LONG",
                                threshold_pos=pivot_threshold_pos,
                                entry_price=entry_price,
                                atr_i=atr_i,
                                swing_positions=high_positions,
                                swing_levels=high_values,
                                target_max_distance_atr_mult=float(target_max_distance_atr_mult),
                            )
                            if target is not None:
                                target_pos, target_level = target
                                long_targets_found += 1
                                rr = (float(target_level) - entry_price) / (entry_price - float(stop_loss))
                                signals.append(
                                    {
                                        "direction": "LONG",
                                        "signal_index": df.index[i],
                                        "entry_price": float(entry_price),
                                        "stop_loss": float(stop_loss),
                                        "target_price": float(target_level),
                                        "rr_ratio": float(rr),
                                        "atr": float(atr_i),
                                        "htf_bias": bias,
                                        "candle_body_ratio": float(candle_body_ratio),
                                        "swing_level_swept": float(swept_level),
                                        "swing_level_swept_index": df.index[swept_pos],
                                        "target_swing_level": float(target_level),
                                        "target_swing_index": df.index[target_pos],
                                        "displacement_atr_mult": float(displacement_atr_mult),
                                        "displacement_bars": int(displacement_bars),
                                        "sweep_buffer_pct": float(sweep_buffer_pct),
                                    }
                                )
                            else:
                                long_targets_none += 1
                                if debug and long_targets_none <= 1:
                                    # Candidate diagnostics: count highs above entry that are within distance.
                                    within_mask = (
                                        (high_positions <= pivot_threshold_pos)
                                        & (high_values > entry_price)
                                        & ((high_values - entry_price) <= float(target_max_distance_atr_mult) * atr_i)
                                    )
                                    num_cands = int(within_mask.sum())
                                    max_cand = float(np.max(high_values[within_mask])) if num_cands > 0 else None
                                    print(
                                        f"DEBUG TARGET LONG NONE: i={i} swept_level={float(swept_level)} entry={entry_price} atr={atr_i} "
                                        f"cands_within={num_cands} max_cand={max_cand}"
                                    )

        # SHORT setup
        if allow_short:
            last_high = _last_position_at_or_before(high_positions, high_values, pivot_threshold_pos)
            if last_high is not None:
                swept_pos, swept_level = last_high
                swept = h > float(swept_level) * (1.0 + float(sweep_buffer_pct))
                # For SHORT, the confirmation candle must reclaim below swept level.
                reversal = (c < o) and (c < float(swept_level)) and (candle_body_ratio > float(body_ratio_threshold))
                if swept and reversal:
                    short_sweeps += 1
                    short_reversals += 1
                    short_sweep_reversals += 1

                    entry_price = float(c)
                    if (float(swept_level) - entry_price) > float(entry_max_distance_atr_mult) * atr_i:
                        short_chase_rejects += 1
                        # Too far from liquidity: do not take this setup.
                    else:
                        stop_loss = float(swept_level) * (1.0 + stop_buffer_pct)
                        if stop_loss > entry_price:
                            short_target_search_attempts += 1
                            target = _find_nearest_opposing_swing(
                                direction="SHORT",
                                threshold_pos=pivot_threshold_pos,
                                entry_price=entry_price,
                                atr_i=atr_i,
                                swing_positions=low_positions,
                                swing_levels=low_values,
                                target_max_distance_atr_mult=float(target_max_distance_atr_mult),
                            )
                            if target is not None:
                                target_pos, target_level = target
                                short_targets_found += 1
                                rr = (entry_price - float(target_level)) / (float(stop_loss) - entry_price)
                                signals.append(
                                    {
                                        "direction": "SHORT",
                                        "signal_index": df.index[i],
                                        "entry_price": float(entry_price),
                                        "stop_loss": float(stop_loss),
                                        "target_price": float(target_level),
                                        "rr_ratio": float(rr),
                                        "atr": float(atr_i),
                                        "htf_bias": bias,
                                        "candle_body_ratio": float(candle_body_ratio),
                                        "swing_level_swept": float(swept_level),
                                        "swing_level_swept_index": df.index[swept_pos],
                                        "target_swing_level": float(target_level),
                                        "target_swing_index": df.index[target_pos],
                                        "displacement_atr_mult": float(displacement_atr_mult),
                                        "displacement_bars": int(displacement_bars),
                                        "sweep_buffer_pct": float(sweep_buffer_pct),
                                    }
                                )
                            else:
                                short_targets_none += 1
                                if debug and short_targets_none <= 1:
                                    within_mask = (
                                        (low_positions <= pivot_threshold_pos)
                                        & (low_values < entry_price)
                                        & ((entry_price - low_values) <= float(target_max_distance_atr_mult) * atr_i)
                                    )
                                    num_cands = int(within_mask.sum())
                                    min_cand = float(np.min(low_values[within_mask])) if num_cands > 0 else None
                                    print(
                                        f"DEBUG TARGET SHORT NONE: i={i} swept_level={float(swept_level)} entry={entry_price} atr={atr_i} "
                                        f"cands_within={num_cands} min_cand={min_cand}"
                                    )

        if debug and i % 200 == 0:
            logger_msg = f"DEBUG strategy progress i={i}/{len(df)} signals={len(signals)}"
            print(logger_msg)

    if debug:
        print(f"DEBUG: Respected low count={int(respected_low.sum())} respected_high count={int(respected_high.sum())}")
        if htf_df is not None:
            # Helps verify HTF bias alignment on the base timeframe.
            unique_biases = base_bias.value_counts().to_dict()
            print(f"DEBUG: HTF bias distribution: {unique_biases}")
        print(
            f"DEBUG: LONG sweeps={long_sweeps} reversals={long_reversals} sweep+rev={long_sweep_reversals} "
            f"target_attempts={long_target_search_attempts} targets_found={long_targets_found} targets_none={long_targets_none} "
            f"chase_rejects={long_chase_rejects}"
        )
        print(
            f"DEBUG: SHORT sweeps={short_sweeps} reversals={short_reversals} sweep+rev={short_sweep_reversals} "
            f"target_attempts={short_target_search_attempts} targets_found={short_targets_found} targets_none={short_targets_none} "
            f"chase_rejects={short_chase_rejects}"
        )
        print(f"DEBUG: Generated {len(signals)} signals")

    return signals