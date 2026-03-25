import os
from dotenv import load_dotenv

load_dotenv()

SYMBOL = ["BTC/USDC:USDC", "ETH/USDC:USDC", "PEPE:USDC/USDC"]    # or ETH/USDT:USDT
TIMEFRAME = "15m"          # entry timeframe
HTF_TIMEFRAME = "1h"       # bias timeframe

RISK_PER_TRADE = 0.01      # 1% of account
LEVERAGE = 10
MIN_RR = 2.0

SWING_LOOKBACK = 20        # candles for swing detection
ATR_PERIOD = 14
SWEEP_BUFFER_PCT = 0.001   # small buffer below/above sweep

# Playbook / strategy tuning defaults
DISPLACEMENT_ATR_MULT = 1.25     # swing "respected" displacement filter (>= 1.0-1.5x ATR)
DISPLACEMENT_BARS = 5           # additional bars after swing touch for displacement confirmation
BODY_RATIO_THRESHOLD = 0.60    # strong reversal candle body ratio
ENTRY_MAX_DISTANCE_ATR_MULT = 0.75   # don't chase too far from swept level
TARGET_MAX_DISTANCE_ATR_MULT = 3.0  # discard signals with far targets
STOP_BUFFER_PCT = SWEEP_BUFFER_PCT     # by default mirror sweep buffer for stop placement

# Backtest / execution realism
TAKER_FEE_PCT = 0.0005          # 0.05% taker fee approximation
SLIPPAGE_PCT = 0.0002           # 0.02% adverse slippage approximation
BACKTEST_YEARS = 2.0
FETCH_LIMIT = 2000             # ccxt limit per request (paging not implemented yet)

# Live trading loop
POLL_INTERVAL_SEC = 10
MAX_DRAWDOWN_PCT = 0.20

USE_TESTNET = True
EXCHANGE = "binance"