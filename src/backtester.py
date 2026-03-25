"""
Backtester: simulate the strategy on historical OHLCV data.

Design goals (aligned with project spec):
- Strategy logic stays in src/strategy.py (pure signal generation).
- Risk sizing stays in src/risk_manager.py (pure planning).
- This backtester is a simple event-driven loop that simulates fills at candle close
  with optional slippage/fees.

Limitations:
- OHLCV backtests can't perfectly model intrabar order sequencing.
  If stop-loss and target are both touched in the same candle, we conservatively assume
  the stop-loss was hit first (worst-case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import SLIPPAGE_PCT, TAKER_FEE_PCT
from src.risk_manager import RiskConfig, plan_trade
from src.strategy import generate_signals
from src.utils import resample_ohlcv, since_ms_for_years, timeframe_to_pandas_rule


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1000.0
    taker_fee_pct: float = float(TAKER_FEE_PCT)
    slippage_pct: float = float(SLIPPAGE_PCT)
    paper: bool = True

    # How many candles to request if fetching from an exchange (approx).
    # ccxt "limit" caps request size; we may need to page in a full version.
    fetch_limit: int = 2000


def _ohlcv_list_to_df(ohlcv: list[list[Any]]) -> pd.DataFrame:
    if not ohlcv:
        raise ValueError("No OHLCV data provided.")
    arr = np.asarray(ohlcv, dtype=object)
    ts = pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "open": arr[:, 1].astype(float),
            "high": arr[:, 2].astype(float),
            "low": arr[:, 3].astype(float),
            "close": arr[:, 4].astype(float),
            "volume": arr[:, 5].astype(float),
        },
        index=ts,
    )
    df = df.sort_index()
    return df


def _adverse_fill_price(*, price: float, side: str, slippage_pct: float) -> float:
    """
    Apply adverse slippage for entry fill.
    side: "BUY" or "SELL" in execution direction.
    """
    if side == "BUY":
        return float(price) * (1.0 + float(slippage_pct))
    if side == "SELL":
        return float(price) * (1.0 - float(slippage_pct))
    raise ValueError("side must be BUY or SELL")


def _adverse_exit_price(*, price: float, direction: str, slippage_pct: float, is_stop: bool) -> float:
    """
    Apply adverse slippage for exit fills.
    direction: "LONG" or "SHORT"
    is_stop: if True, exit due to stop-loss.
    """
    direction = direction.upper()
    if is_stop:
        if direction == "LONG":
            return float(price) * (1.0 - float(slippage_pct))
        if direction == "SHORT":
            return float(price) * (1.0 + float(slippage_pct))
    else:
        # target
        if direction == "LONG":
            return float(price) * (1.0 - float(slippage_pct))
        if direction == "SHORT":
            return float(price) * (1.0 + float(slippage_pct))
    raise ValueError("Invalid direction / is_stop combination")


def _apply_fees(*, quantity: float, entry_price: float, exit_price: float, fee_pct: float) -> float:
    entry_fee = float(quantity) * float(entry_price) * float(fee_pct)
    exit_fee = float(quantity) * float(exit_price) * float(fee_pct)
    return entry_fee + exit_fee


def backtest_symbol(
    *,
    symbol: str,
    timeframe: str,
    htf_timeframe: str,
    years: float,
    exchange_client: Any,
    risk_config: RiskConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
) -> dict[str, Any]:
    """
    Fetch historical OHLCV and run a backtest.

    exchange_client: expected to provide:
      - fetch_ohlcv(symbol, timeframe, since_ms, limit)
    """
    since_ms = since_ms_for_years(years)
    ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe, since_ms=since_ms, limit=backtest_config.fetch_limit)
    df = _ohlcv_list_to_df(ohlcv)

    # Resample base timeframe to HTF for market-structure bias.
    htf_rule = timeframe_to_pandas_rule(htf_timeframe)
    base_rule = timeframe_to_pandas_rule(timeframe)
    # base_rule is unused currently but kept for readability if paging requires alignment.
    _ = base_rule
    htf_df = resample_ohlcv(df, rule=htf_rule)

    equity = float(backtest_config.initial_equity)
    equity_curve: list[float] = []
    trade_journal: list[dict[str, Any]] = []

    # Generate signals once, then simulate.
    # This avoids recomputing the strategy on each candle.
    signals = generate_signals(df, htf_df)
    # Sort signals by their candle time.
    signals = sorted(signals, key=lambda s: s["signal_index"])

    open_trade: Optional[dict[str, Any]] = None
    current_signal_ptr = 0

    # Walk candle-by-candle to check stop/target hits.
    signal_times = [s["signal_index"] for s in signals]
    for ts, row in df.iterrows():
        # Mark-to-market each candle (simple mark-to-close if open, else equity).
        equity_mark = equity
        if open_trade is not None:
            equity_mark = equity + float(open_trade["qty"]) * (float(row["close"]) - float(open_trade["entry_exec_price"]))
            if open_trade["direction"] == "SHORT":
                equity_mark = equity + float(open_trade["qty"]) * (float(open_trade["entry_exec_price"]) - float(row["close"]))
        equity_curve.append(equity_mark)

        # Open a trade if we have a signal at this exact candle.
        while current_signal_ptr < len(signals) and signal_times[current_signal_ptr] == ts:
            if open_trade is None:
                sig = signals[current_signal_ptr]
                # Risk sizing based on current equity at entry.
                planned = plan_trade(signal=sig, equity=equity, risk_config=risk_config)
                entry_exec = _adverse_fill_price(
                    price=planned["entry_price"],
                    side=planned["side"],
                    slippage_pct=backtest_config.slippage_pct,
                )
                open_trade = {
                    "direction": planned["direction"],
                    "qty": float(planned["quantity"]),
                    "entry_price_theoretical": float(planned["entry_price"]),
                    "entry_exec_price": float(entry_exec),
                    "stop_loss": float(planned["stop_loss"]),
                    "target_price": float(planned["target_price"]),
                    "signal": sig,
                    "entry_ts": ts,
                }
            current_signal_ptr += 1

        if open_trade is None:
            continue

        # Check stop/target hits using candle extremes.
        direction = open_trade["direction"]
        stop_loss = open_trade["stop_loss"]
        target_price = open_trade["target_price"]
        candle_high = float(row["high"])
        candle_low = float(row["low"])

        hit_stop = False
        hit_target = False

        if direction == "LONG":
            hit_stop = candle_low <= stop_loss
            hit_target = candle_high >= target_price
        else:
            hit_stop = candle_high >= stop_loss
            hit_target = candle_low <= target_price

        if not hit_stop and not hit_target:
            continue

        # Worst-case: if both hit in same candle, assume stop-loss first.
        is_stop_exit = hit_stop
        exit_reason = "STOP_LOSS" if is_stop_exit else "TAKE_PROFIT"

        if is_stop_exit:
            exit_theoretical = stop_loss
        else:
            exit_theoretical = target_price

        exit_exec = _adverse_exit_price(
            price=exit_theoretical,
            direction=direction,
            slippage_pct=backtest_config.slippage_pct,
            is_stop=is_stop_exit,
        )

        # Compute net PnL and apply fees.
        entry_exec = float(open_trade["entry_exec_price"])
        qty = float(open_trade["qty"])

        if direction == "LONG":
            gross_pnl = qty * (exit_exec - entry_exec)
        else:
            gross_pnl = qty * (entry_exec - exit_exec)

        fees = _apply_fees(
            quantity=qty,
            entry_price=entry_exec,
            exit_price=exit_exec,
            fee_pct=backtest_config.taker_fee_pct,
        )

        net_pnl = gross_pnl - fees
        equity += net_pnl

        trade_journal.append(
            {
                "entry_ts": open_trade["entry_ts"],
                "exit_ts": ts,
                "direction": direction,
                "entry_price": open_trade["entry_price_theoretical"],
                "entry_exec_price": entry_exec,
                "exit_price": exit_theoretical,
                "exit_exec_price": exit_exec,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "exit_reason": exit_reason,
                "quantity": qty,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "net_pnl": net_pnl,
                "rr_ratio": open_trade["signal"].get("rr_ratio"),
                "atr": open_trade["signal"].get("atr"),
                "htf_bias": open_trade["signal"].get("htf_bias"),
            }
        )

        open_trade = None

    equity_arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = peak - equity_arr
    max_drawdown = float(np.max(drawdown)) if drawdown.size else 0.0

    journal_df = pd.DataFrame(trade_journal)
    wins = int((journal_df["net_pnl"] > 0).sum()) if not journal_df.empty else 0
    losses = int((journal_df["net_pnl"] < 0).sum()) if not journal_df.empty else 0
    total_profit = float(journal_df.loc[journal_df["net_pnl"] > 0, "net_pnl"].sum()) if not journal_df.empty else 0.0
    total_loss = float(journal_df.loc[journal_df["net_pnl"] < 0, "net_pnl"].sum()) if not journal_df.empty else 0.0
    profit_factor = float(total_profit / abs(total_loss)) if total_loss != 0 else None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "htf_timeframe": htf_timeframe,
        "years": years,
        "initial_equity": float(backtest_config.initial_equity),
        "final_equity": float(equity),
        "max_drawdown": max_drawdown,
        "num_trades": int(len(trade_journal)),
        "wins": wins,
        "losses": losses,
        "profit_factor": profit_factor,
        "equity_curve": equity_curve,
        "trade_journal": journal_df,
    }

