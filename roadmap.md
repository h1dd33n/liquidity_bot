# Trading Bot Roadmap

## Current State (end-to-end pipeline works)
- `src/strategy.py`
  - Generates liquidity-sweep/grab style signals.
  - Uses “nearest previous respected opposing liquidity swing” as the target.
  - Does **not** attempt to predict/guarantee RR inside the signal generator (RR is validated in `risk_manager.plan_trade`).
  - Includes debug counters/logging to show sweeps/reversals and target search outcomes.
- `src/backtester.py`
  - Fetches OHLCV with **pagination**, so `--years` actually pulls enough candles.
  - Runs `strategy.generate_signals()` once, simulates trades candle-by-candle, writes metrics + (when non-empty) a trade journal.
  - Skips signals when `risk_manager.plan_trade()` rejects them due to RR being below `MIN_RR` instead of crashing.
- `run_backtest.py`
  - Prints a compact summary instead of dumping large equity curves to stdout.
- Observed outcome
  - Trades now occur (so the pipeline is functioning), but **all trades are losing** under the current execution/exit assumptions.

## Near-Term Goals (get non-zero win rate)
1. Verify execution model correctness
   - Entry slippage/adverse fill vs stop/target placement: ensure stop/target are consistent with the *actual* filled entry price.
   - Candle hit ordering: when both stop and target are inside the same candle range, confirm whether “stop-first” is appropriate for our order type and exchange mechanics.
   - Stop/target fill price selection realism (market assumptions vs limit/stop-market behavior).
2. Make RR gating observable
   - Add per-signal debug to record computed RR and the planned levels before the backtester simulates fills.
   - Confirm that `risk_manager` RR >= `MIN_RR` actually corresponds to what the backtester uses when checking `high/low` triggers.
3. Reduce noisy signal frequency while debugging
   - Use tighter sampling/backtest duration (e.g., `--years 0.2`) and track specific losing examples.
   - Export a small “diagnostic subset” of trade rows (first N wins/losses) for fast inspection.

## Mid-Term Goals (make it “professional-grade”)
4. Unit test coverage for signal geometry
   - Tests for:
     - sweep condition correctness
     - reversal confirmation (body ratio)
     - respected swing detection (ATR displacement)
     - target selection (nearest previous opposing swing)
5. Unit test coverage for risk manager
   - RR computation consistency with strategy signal levels.
   - Quantity rounding respecting min_qty/step_size.
   - Stop/target distance sanity (stop < entry < target for LONG, etc.).
6. Backtester realism improvements
   - Model entry/exit timing more explicitly (e.g., entry executed on next candle open vs same-candle).
   - Adjust stop/target based on actual executed entry price (if slippage is applied).
   - Optionally simulate partial fills if your order model requires it.
7. Live/paper trading parity
   - Ensure live/paper uses the same assumptions about order types as the backtester.
   - Add “paper mode” journaling that matches the backtester trade journal schema.

## Long-Term Goals (robust trading system)
8. Parameter calibration workflow
   - Create a repeatable process to tune:
     - `displacement_atr_mult`
     - `displacement_bars`
     - `body_ratio_threshold`
     - `sweep_buffer_pct`
     - `entry_max_distance_atr_mult`
     - `target_max_distance_atr_mult`
   - Track results per run (CSV/JSON) with seedable configs.
9. Monitoring & safety
   - Max drawdown shutdown (already scaffolded).
   - Alerting/logging for repeated stop-outs and order failures.
10. Performance
   - Add cached OHLCV fetching (CSV cache exists in `src/utils.py`).
   - Add faster analytics (optional) while preserving correctness.

## What we will do next
1. Inspect `src/risk_manager.py` and `src/backtester.py` to find the reason win rate is 0%.
2. Pull a small slice of the latest `*_trades.csv` to compare:
   - planned entry/stop/target
   - executed entry (after adverse fill)
   - which exit got triggered and why.
3. Apply the smallest code changes necessary to fix the execution/exit mismatch.

