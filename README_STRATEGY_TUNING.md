# README_STRATEGY_TUNING.md

## What this is
This repo’s liquidity-grab strategy is implemented as pure signal generation in `src/strategy.py`:
`generate_signals(df, htf_df=None, ...) -> list[dict]`.

This document lists the tuning knobs that affect signal frequency vs selectivity, so you can iterate safely.

## Strategy tuning knobs (main)
These correspond to `generate_signals()` parameters in `src/strategy.py`:

### Respected swing / displacement filter
- `displacement_atr_mult` (float, default `1.0`)
  - A swing is only considered “respected” if price moves away by at least:
    `displacement_atr_mult * ATR` after the swing is touched.
- `displacement_bars` (int or None, default `5`)
  - Look-forward bars used to measure displacement strength.
  - If set to `None`, it falls back to `5`.

### Liquidity sweep threshold
- `sweep_buffer_pct` (float, default `0.001`)
  - LONG sweep: `low < swing_low * (1 - sweep_buffer_pct)`
  - SHORT sweep: `high > swing_high * (1 + sweep_buffer_pct)`

### Strong reversal confirmation
- `body_ratio_threshold` (float, default `0.55`)
  - Confirmation requires candle body strength:
    `abs(close-open) / (high-low) > body_ratio_threshold`

### Entry/target distance constraints (anti-chasing)
- `entry_max_distance_atr_mult` (float, default `1.5`)
  - Rejects entries too far from the swept liquidity level:
    - LONG: `(entry - swept_level) > entry_max_distance_atr_mult * ATR`
    - SHORT: `(swept_level - entry) > entry_max_distance_atr_mult * ATR`
- `target_max_distance_atr_mult` (float, default `5.0`)
  - Rejects signals if the nearest opposing liquidity target is too far:
    `target_dist > target_max_distance_atr_mult * ATR`

### Minimum reward-to-risk
- `min_rr` (float, default `2.0`)
  - Signals are discarded if computed `RR < min_rr`.
  - When you’re ready for stricter playbook compliance, increase back toward `2.0`.

## HTF bias gate
Behavior:
- HTF bias `"neutral"` is treated as allowing both LONG and SHORT gating only when
  `allow_neutral_htf_in_backtest=True` (backtest calibration).
- For live trading, keep `allow_neutral_htf_in_backtest=False` so bias is required.

If you want stricter structure-first compliance later, change those conditions back to:
- LONG only when HTF bias is `"bullish"`
- SHORT only when HTF bias is `"bearish"`

## Recommended iteration ladder
1. Start with current calibration defaults (higher signal frequency).
2. Once you confirm non-zero trades and acceptable expectancy, tighten gradually:
   - `min_rr` toward `2.0`
   - `body_ratio_threshold` toward `0.60`
   - `displacement_atr_mult` toward `1.0-1.25`
   - `entry_max_distance_atr_mult` toward `0.75`
   - `target_max_distance_atr_mult` toward `3.0`

## Where to look in code
- `src/strategy.py` → `generate_signals()` and `_compute_respected_swings()`

