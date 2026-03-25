"""
Exchange client wrapper (ccxt).

Goals:
- Keep live_bot.py clean by centralizing:
  - testnet vs live
  - leverage/margin configuration
  - symbol metadata (min_qty, step_size, tick_size)
  - basic trading primitives (price/balance/positions/orders)

This file is exchange-agnostic in interface, but currently tuned for Binance Futures
where stop-loss / take-profit market orders are supported.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import ccxt

from config.settings import LEVERAGE, USE_TESTNET, EXCHANGE, MIN_RR, RISK_PER_TRADE
from src.risk_manager import RiskConfig


class ExchangeClient:
    def __init__(
        self,
        *,
        exchange_id: str = EXCHANGE,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_testnet: bool = USE_TESTNET,
        leverage: float = LEVERAGE,
        margin_mode: str = "isolated",
    ) -> None:
        self.exchange_id = exchange_id
        self.use_testnet = bool(use_testnet)
        self.leverage = float(leverage)
        self.margin_mode = margin_mode

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: Any = exchange_class(
            {
                "apiKey": api_key or "",
                "secret": api_secret or "",
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

        # Configure sandbox/testnet mode if supported by ccxt.
        if self.use_testnet and hasattr(self.exchange, "set_sandbox_mode"):
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                # We'll continue; some users configure IP allowlists for the real endpoints anyway.
                pass

        # Best-effort URL override (primarily for Binance testnet futures).
        if self.use_testnet and self.exchange_id == "binance":
            try:
                # Binance futures testnet base
                base = "https://testnet.binancefuture.com"
                self.exchange.urls = {
                    **getattr(self.exchange, "urls", {}),
                    "api": {**getattr(self.exchange, "urls", {}).get("api", {}), "public": base, "private": base},
                }
            except Exception:
                pass

        self.markets_loaded = False

    def load_markets(self) -> None:
        if not self.markets_loaded:
            self.exchange.load_markets()
            self.markets_loaded = True

    def configure_leverage_and_margin(self, symbol: str) -> None:
        """
        Set leverage and margin mode (isolated recommended).

        Note: ccxt method coverage varies; we try common methods and swallow failures.
        """
        self.load_markets()

        # Leverage
        if self.leverage:
            try:
                if hasattr(self.exchange, "set_leverage"):
                    self.exchange.set_leverage(int(self.leverage), symbol)
            except Exception:
                pass

        # Margin mode
        if self.margin_mode:
            try:
                if hasattr(self.exchange, "set_margin_mode"):
                    # Some exchanges expect margin mode strings like 'isolated' / 'cross'
                    self.exchange.set_margin_mode(self.margin_mode, symbol)
            except Exception:
                pass

    def get_symbol_constraints(self, symbol: str) -> RiskConfig:
        """
        Convert ccxt market metadata into RiskConfig constraints.
        """
        self.load_markets()
        market = self.exchange.market(symbol)

        min_qty = None
        step_size = None
        tick_size = None

        try:
            min_qty = market.get("limits", {}).get("amount", {}).get("min")
        except Exception:
            min_qty = None

        # step_size: from precision (decimals) -> quantization step
        try:
            amt_precision = market.get("precision", {}).get("amount")
            if amt_precision is not None:
                step_size = float(10 ** (-int(amt_precision)))
        except Exception:
            step_size = None

        try:
            price_precision = market.get("precision", {}).get("price")
            if price_precision is not None:
                tick_size = float(10 ** (-int(price_precision)))
        except Exception:
            tick_size = None

        return RiskConfig(
            risk_per_trade=float(RISK_PER_TRADE),
            leverage=self.leverage,
            min_rr=float(MIN_RR),
            min_qty=min_qty,
            step_size=step_size,
        )

    def get_current_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        if price is None:
            price = ticker.get("close")
        return float(price)

    def get_balance_equity(self, quote_currency: Optional[str] = None) -> float:
        """
        Fetch account equity approximation from futures balance.
        """
        balance = self.exchange.fetch_balance()

        if quote_currency and quote_currency in balance:
            total = balance[quote_currency].get("total")
            if total is not None:
                return float(total)

        # Fallback: sum all 'total' balances.
        total_equity = 0.0
        for _, v in balance.items():
            if isinstance(v, dict) and "total" in v and v["total"] is not None:
                total_equity += float(v["total"])
        return float(total_equity)

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        if not hasattr(self.exchange, "fetch_positions"):
            return []
        try:
            positions = self.exchange.fetch_positions([symbol] if symbol else None)
            # Normalize into list of dicts
            return [p for p in positions if p]
        except Exception:
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str, *, since_ms: Optional[int] = None, limit: int = 1500) -> list[list[Any]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

    def place_trade(self, *, symbol: str, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Place market entry and reduce-only SL/TP orders.
        """
        self.configure_leverage_and_margin(symbol)

        side = str(plan["side"]).upper()  # BUY/SELL
        quantity = float(plan["quantity"])
        entry_price = float(plan["entry_price"])
        stop_loss = float(plan["stop_loss"])
        target_price = float(plan["target_price"])

        # Entry
        entry_order = self.exchange.create_order(
            symbol=symbol,
            type="MARKET",
            side="buy" if side == "BUY" else "sell",
            amount=quantity,
        )

        # Protective orders (opposite side, reduceOnly)
        reduce_side = "sell" if side == "BUY" else "buy"
        common_params = {"reduceOnly": True}

        sl_order = None
        tp_order = None

        # Stop-loss
        try:
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=reduce_side,
                amount=quantity,
                params={**common_params, "stopPrice": stop_loss},
            )
        except Exception:
            # Fallback to stop-limit if STOP_MARKET isn't supported
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_LIMIT",
                side=reduce_side,
                amount=quantity,
                params={**common_params, "stopPrice": stop_loss, "price": stop_loss},
            )

        # Take-profit
        try:
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=reduce_side,
                amount=quantity,
                params={**common_params, "stopPrice": target_price},
            )
        except Exception:
            # Fallback to take-profit limit
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_LIMIT",
                side=reduce_side,
                amount=quantity,
                params={**common_params, "stopPrice": target_price, "price": target_price},
            )

        return {
            "entry_order": entry_order,
            "sl_order": sl_order,
            "tp_order": tp_order,
            "planned_entry_price": entry_price,
        }

