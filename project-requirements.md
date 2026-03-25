# Project Requirements: crypto-liquidity-grab-bot

## Project Overview
This is a **professional-grade Python trading bot** focused exclusively on the **Liquidity Trap / Liquidity Grab strategy** taught by Marco Trades on the Chart Fanatics YouTube channel (videos such as "STEAL This EASY Liquidity TRAP Trading Strategy" and the official Liquidity Strategy Playbook).

The bot automates catching sharp reversals after clean liquidity sweeps on **crypto perpetual futures** (primarily BTC/USDT:USDT and ETH/USDT:USDT on Binance Futures, with easy support for Bybit).

**Core Philosophy (direct from Marco Trades):**
- "Buy below lows, sell above highs" — but **only after** the market has swept (taken out) a respected swing high or low that previously moved strongly away from it, creating resting liquidity.
- Never enter before the liquidity grab (no predictive entries).
- Wait for the trap to spring, then trade the reversal with tight risk.
- Target the next opposing liquidity pool (previous respected swing high for longs, swing low for shorts).
- Use liquidity itself to determine directional bias.
- Strict rule: No trade without a clear liquidity sweep + reversal confirmation.

The bot must be **modular, well-documented, safe-first** (paper trading by default), and built for rigorous backtesting before any live capital is risked.

## Exact Strategy Rules to Implement (Non-Negotiable)

### 1. Swing Detection
- Identify "respected" swing highs and lows using left/right bars (default 5–10) or fractal logic.
- A swing is "respected" if price moved strongly away from it (minimum distance filter using ATR).

### 2. Liquidity Sweep Detection
- **Long Setup (Bullish Liquidity Grab)**:
  - Price sweeps **below** a previous respected swing low (takes buy-side stops/liquidity).
  - Confirmation: Strong reversal candle (bullish body > 60% of range) **or** break of internal structure higher.
  - Enter **long near or just above** the swept low (tight entry).
- **Short Setup (Bearish Liquidity Grab)**:
  - Price sweeps **above** a previous respected swing high (takes sell-side stops).
  - Confirmation: Strong bearish reversal candle.
  - Enter **short near or just below** the swept high.

### 3. Risk & Trade Management
- Stop-loss: Just beyond the extreme of the sweep (small buffer, e.g., 0.1% or 1–2 ticks).
- Target: Next opposing liquidity pool (previous respected swing high for longs, swing low for shorts). Minimum R:R = 2.0.
- Position sizing: Risk exactly `RISK_PER_TRADE` % of current equity per trade (default 1%).
- Leverage: Configurable (default 10x, isolated mode recommended).
- Max concurrent positions: 1 (to keep risk controlled).
- Optional trailing stop or partial exits at liquidity levels.

### 4. Filters (Mandatory for Edge Preservation)
- Higher-timeframe bias (1h or 4h): Only take longs if HTF structure is bullish (higher highs/higher lows or EMA trend). Same for shorts.
- Minimum swing strength (ATR multiplier).
- Avoid extremely low-volume periods or insane volatility spikes.
- Time/session filter if desired (e.g., higher activity hours).

## Technical Architecture & Backend Details

**Language & Stack**
- Python 3.11+
- Primary exchange library: **ccxt** (pro version for WebSocket support)
- Data handling: pandas + pandas_ta (or TA-Lib)
- Backtesting: vectorbt (preferred for speed) or backtrader
- Logging: loguru (structured, colorful, file + console)
- Config: pydantic-settings or dotenv + settings.py
- Async support: asyncio for live WebSocket loop

**Project Structure (Must Follow Exactly)**
crypto-liquidity-grab-bot/
├── project-requirements.md          # ← This file (single source of truth)
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py                  # All parameters, symbols, risk settings
├── data/                            # Cached OHLCV data (gitignored)
├── logs/                            # Daily log files
├── backtests/                       # Results, equity curves, trade journals
├── src/
│   ├── init.py
│   ├── strategy.py                  # Pure strategy logic: swings, sweeps, signals
│   ├── risk_manager.py              # Position sizing, stop/target calculation
│   ├── backtester.py                # Full backtest engine with realistic fees/slippage
│   ├── live_bot.py                  # Async main loop with WebSocket candle feed
│   ├── exchange_client.py           # ccxt wrapper (testnet/real, leverage, orders)
│   ├── utils.py                     # Helpers (ATR, structure detection, etc.)
│   └── notifications.py             # Optional Telegram alerts
├── run_backtest.py                  # Entry point for backtesting
├── run_live.py                      # Entry point for paper or live trading
└── tests/                           # Unit tests for strategy logic (optional but recommended)


**Key Implementation Details Cursor Must Respect**
- All strategy logic must stay in `src/strategy.py` as pure functions that take pandas DataFrames and return signals. No exchange calls inside strategy.
- Backtester must simulate realistic Binance Futures conditions:
  - Taker fees ≈ 0.04–0.06%
  - Slippage model (configurable)
  - Funding rate impact (optional)
  - Leverage and isolated margin
- Live bot must:
  - Use ccxt.pro WebSocket for real-time 5m/15m candles
  - Fetch higher-timeframe bias periodically
  - Support dry-run / paper mode flag (`--paper` or `USE_TESTNET=True`)
  - Implement graceful shutdown and emergency stop on large drawdown
- Safety defaults:
  - Paper/testnet mode is **always** the default until explicitly changed
  - Hard cap on risk per trade
  - No martingale or grid logic — one high-quality setup at a time

**Performance & Validation Requirements**
- Backtest at least 2 years of 5m/15m data for BTC and ETH.
- Report: win rate, profit factor, max drawdown, Sharpe ratio, number of trades, expectancy.
- Only move to paper trading after backtest shows positive expectancy after fees.
- Paper trade minimum 2–4 weeks before considering real capital.

**Extensibility**
- Easy to add more filters (FVG, order blocks, volume confirmation, news avoidance).
- Easy to switch symbols or timeframes via config.
- Easy to add Bybit support later.

**Non-Goals (Do NOT Implement)**
- No machine learning or parameter optimization that leads to curve-fitting.
- No multi-strategy orchestration yet.
- No front-end/UI (keep it CLI + logs + optional Telegram).

This document is the **authoritative specification**. Any code generated must strictly align with the strategy rules, architecture, and safety principles outlined here. If something is ambiguous, refer back to Marco Trades' core principle: "Wait for the liquidity to be taken, then trade the reversal with tight risk toward the next liquidity pool."

Last updated: March 2026