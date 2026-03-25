"""
Live (or paper) trading loop.

This implementation favors correctness of integration over maximum efficiency:
- polls for new candles (REST) by default
- falls back gracefully if ccxt.pro is not available
- on each new candle close:
    generate_signals -> plan_trade -> (optional) place_trade

Position monitoring and advanced order management can be layered on later.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import MAX_DRAWDOWN_PCT, POLL_INTERVAL_SEC, TAKER_FEE_PCT
from src.exchange_client import ExchangeClient
from src.risk_manager import RiskConfig, plan_trade
from src.strategy import generate_signals
from src.utils import resample_ohlcv, timeframe_to_pandas_rule


@dataclass(frozen=True)
class LiveBotConfig:
    paper: bool = True
    poll_interval_sec: int = int(POLL_INTERVAL_SEC)
    lookback_candles: int = 500
    taker_fee_pct: float = float(TAKER_FEE_PCT)  # informational for journal only
    max_drawdown_pct: float = float(MAX_DRAWDOWN_PCT)  # emergency stop threshold


def _ohlcv_to_df(ohlcv: list[list[Any]]) -> pd.DataFrame:
    if not ohlcv:
        raise ValueError("No OHLCV returned.")
    arr = np.asarray(ohlcv, dtype=object)
    ts = pd.to_datetime(arr[:, 0].astype("int64"), unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "open": arr[:, 1].astype(float),
            "high": arr[:, 2].astype(float),
            "low": arr[:, 3].astype(float),
            "close": arr[:, 4].astype(float),
            "volume": arr[:, 5].astype(float),
        },
        index=ts,
    ).sort_index()
    return df


async def run_live_loop(
    *,
    exchange_client: ExchangeClient,
    symbol: str,
    timeframe: str,
    htf_timeframe: str,
    risk_config: RiskConfig,
    live_config: LiveBotConfig = LiveBotConfig(),
) -> None:
    """
    Main trading loop.
    """
    base_rule = timeframe_to_pandas_rule(timeframe)
    htf_rule = timeframe_to_pandas_rule(htf_timeframe)
    _ = base_rule

    equity_peak: Optional[float] = None
    last_candle_ts: Optional[pd.Timestamp] = None
    current_position_open = False

    logger.info(f"Starting live loop for {symbol} timeframe={timeframe} htf={htf_timeframe} paper={live_config.paper}")

    while True:
        try:
            # Fetch recent candles (polling).
            # ccxt 'fetch_ohlcv' returns last N if since_ms is omitted; not guaranteed across exchanges.
            ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe, since_ms=None, limit=live_config.lookback_candles)
            df = _ohlcv_to_df(ohlcv)
            htf_df = resample_ohlcv(df, rule=htf_rule)

            latest_ts = df.index[-1]
            if last_candle_ts is not None and latest_ts <= last_candle_ts:
                await asyncio.sleep(live_config.poll_interval_sec)
                continue
            last_candle_ts = latest_ts

            # Emergency stop via equity drawdown.
            equity = exchange_client.get_balance_equity()
            if equity_peak is None:
                equity_peak = equity
            else:
                dd = (equity_peak - equity) / equity_peak if equity_peak > 0 else 0.0
                if dd >= live_config.max_drawdown_pct:
                    logger.error(f"Max drawdown reached: {dd:.2%}. Stopping bot.")
                    return

            # Refresh open position state.
            positions = exchange_client.get_open_positions(symbol)
            current_position_open = any(
                (p.get("contracts") not in (None, 0, "0"))
                for p in positions
            )

            if current_position_open:
                await asyncio.sleep(live_config.poll_interval_sec)
                continue

            signals = generate_signals(df, htf_df)
            if not signals:
                await asyncio.sleep(live_config.poll_interval_sec)
                continue

            # Take the last signal if it matches the latest candle close time.
            signals_sorted = sorted(signals, key=lambda s: s.get("signal_index"))
            last_sig = signals_sorted[-1]
            if last_sig.get("signal_index") != latest_ts:
                await asyncio.sleep(live_config.poll_interval_sec)
                continue

            plan = plan_trade(signal=last_sig, equity=equity, risk_config=risk_config)
            logger.info(
                f"Signal: {plan['direction']} qty={plan['quantity']:.6f} entry={plan['entry_price']} "
                f"SL={plan['stop_loss']} TP={plan['target_price']} RR={plan['rr_ratio']:.2f}"
            )

            if live_config.paper:
                logger.info("Paper mode enabled; not placing orders.")
                await asyncio.sleep(live_config.poll_interval_sec)
                continue

            await asyncio.to_thread(exchange_client.place_trade, symbol=symbol, plan=plan)
            await asyncio.sleep(live_config.poll_interval_sec)
        except Exception as e:
            logger.exception(f"Live loop error: {e}")
            await asyncio.sleep(live_config.poll_interval_sec)

