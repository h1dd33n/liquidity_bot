from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from config.settings import HTF_TIMEFRAME, SYMBOL, TIMEFRAME
from src.backtester import BacktestConfig, backtest_symbol
from src.exchange_client import ExchangeClient
from src.risk_manager import RiskConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=SYMBOL[0])
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME)
    parser.add_argument("--htf_timeframe", type=str, default=HTF_TIMEFRAME)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--initial_equity", type=float, default=1000.0)
    parser.add_argument("--taker_fee_pct", type=float, default=0.0005)
    parser.add_argument("--slippage_pct", type=float, default=0.0002)
    args = parser.parse_args()

    client = ExchangeClient()

    # Build risk config from exchange constraints when possible.
    try:
        symbol_constraints = client.get_symbol_constraints(args.symbol)
        risk_config = RiskConfig(
            risk_per_trade=symbol_constraints.risk_per_trade,
            leverage=symbol_constraints.leverage,
            min_rr=symbol_constraints.min_rr,
            min_qty=symbol_constraints.min_qty,
            step_size=symbol_constraints.step_size,
        )
    except Exception:
        # Fallback: let risk_manager compute without exchange quantization constraints.
        risk_config = RiskConfig()

    cfg = BacktestConfig(
        initial_equity=args.initial_equity,
        taker_fee_pct=args.taker_fee_pct,
        slippage_pct=args.slippage_pct,
    )

    logger.info(
        f"Running backtest symbol={args.symbol} timeframe={args.timeframe} htf={args.htf_timeframe} years={args.years}"
    )
    result = backtest_symbol(
        symbol=args.symbol,
        timeframe=args.timeframe,
        htf_timeframe=args.htf_timeframe,
        years=args.years,
        exchange_client=client,
        risk_config=risk_config,
        backtest_config=cfg,
    )

    out_dir = os.path.join("backtests", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)

    trade_journal = result["trade_journal"]
    if isinstance(trade_journal, pd.DataFrame) and not trade_journal.empty:
        trade_path = os.path.join(out_dir, f"{args.symbol.replace('/', '-')}_trades.csv")
        trade_journal.to_csv(trade_path, index=False)

    metrics_path = os.path.join(out_dir, f"{args.symbol.replace('/', '-')}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in result.items():
            if k in {"equity_curve", "trade_journal"}:
                continue
            f.write(f"{k}: {v}\n")

    logger.info(
        "Backtest complete: final_equity={final_equity} max_drawdown={max_drawdown} num_trades={num_trades}".format(
            final_equity=result["final_equity"], max_drawdown=result["max_drawdown"], num_trades=result["num_trades"]
        )
    )

    print(result)


if __name__ == "__main__":
    main()

