"""
Project utilities (exchange-agnostic).

Keep this file focused on pure helpers used by:
- backtester.py
- live_bot.py
- exchange_client.py (for data formatting/caching only)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df is sorted by a DatetimeIndex.
    Expects either:
    - an existing DatetimeIndex, or
    - a column named 'timestamp' (ms) or 'time' (ms/iso).
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        out = out.set_index("timestamp")
        return out.sort_index()

    if "time" in df.columns:
        out = df.copy()
        # Best-effort parsing (ms epoch or ISO-like strings).
        if pd.api.types.is_numeric_dtype(out["time"]):
            out["time"] = pd.to_datetime(out["time"], unit="ms", utc=True)
        else:
            out["time"] = pd.to_datetime(out["time"], utc=True)
        out = out.set_index("time")
        return out.sort_index()

    raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp'/'time' column.")


def timeframe_to_pandas_rule(timeframe: str) -> str:
    """
    Convert common ccxt timeframe strings to pandas resample rules.
    Examples:
      '15m' -> '15min'
      '1h'  -> '1h'
      '4h'  -> '4h'
      '1d'  -> '1D'
    """
    m = re.fullmatch(r"(\d+)\s*([mhdw])", timeframe.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
    qty = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return f"{qty}min"
    if unit == "h":
        return f"{qty}h"
    if unit == "d":
        return f"{qty}D"
    if unit == "w":
        return f"{qty}W"
    raise ValueError(f"Unsupported timeframe unit: {unit}")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV with standard aggregation:
      open=first, high=max, low=min, close=last, volume=sum
    """
    df = ensure_datetime_index(df)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for resample: {missing}")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(ohlc_dict).dropna(subset=["open", "high", "low", "close"])
    return out


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def since_ms_for_years(years: float) -> int:
    days = int(years * 365.25)
    return int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_path(data_dir: str, *, symbol: str, timeframe: str, since_ms: int) -> str:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return os.path.join(data_dir, f"{safe_symbol}__{timeframe}__since{since_ms}.csv")


def save_ohlcv_csv(df: pd.DataFrame, path: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex when saving OHLCV cache.")
    out = out.copy()
    out["timestamp"] = (out.index.view("i8") // 10**6).astype("int64")
    out = out.reset_index(drop=True)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    out = out[cols]
    out.to_csv(path, index=False)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Invalid cache CSV: missing 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]

