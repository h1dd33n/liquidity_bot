from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from config.settings import HTF_TIMEFRAME, SYMBOL, TIMEFRAME
from src.exchange_client import ExchangeClient
from src.live_bot import LiveBotConfig, run_live_loop
from src.risk_manager import RiskConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=SYMBOL[0])
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME)
    parser.add_argument("--htf_timeframe", type=str, default=HTF_TIMEFRAME)
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--no-paper", dest="paper", action="store_false")
    parser.add_argument("--poll_interval_sec", type=int, default=10)
    args = parser.parse_args()

    client = ExchangeClient()

    # Risk config from exchange constraints when possible.
    try:
        sc = client.get_symbol_constraints(args.symbol)
        risk_config = RiskConfig(
            risk_per_trade=sc.risk_per_trade,
            leverage=sc.leverage,
            min_rr=sc.min_rr,
            min_qty=sc.min_qty,
            step_size=sc.step_size,
        )
    except Exception:
        risk_config = RiskConfig()

    live_cfg = LiveBotConfig(paper=args.paper, poll_interval_sec=args.poll_interval_sec)
    logger.info(f"Launching live loop paper={live_cfg.paper}")
    asyncio.run(
        run_live_loop(
            exchange_client=client,
            symbol=args.symbol,
            timeframe=args.timeframe,
            htf_timeframe=args.htf_timeframe,
            risk_config=risk_config,
            live_config=live_cfg,
        )
    )


if __name__ == "__main__":
    main()

