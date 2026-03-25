"""
Risk manager: convert strategy signals into trade sizing + order parameters.

This module is intentionally exchange-agnostic and has no ccxt/binance calls.
It assumes linear (USDT/USDC-margined) futures where PnL is approximately:
  pnl ~= quantity * (exit_price - entry_price)
so the loss at stop-loss is:
  loss ~= quantity * (entry_price - stop_loss)   (LONG)
  loss ~= quantity * (stop_loss - entry_price)   (SHORT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _direction_to_side(direction: str) -> str:
    if direction == "LONG":
        return "BUY"
    if direction == "SHORT":
        return "SELL"
    raise ValueError("direction must be 'LONG' or 'SHORT'")


def validate_signal(signal: dict[str, Any]) -> None:
    required_keys = (
        "direction",
        "entry_price",
        "stop_loss",
        "target_price",
        "rr_ratio",
        "atr",
        "htf_bias",
        "candle_body_ratio",
        "swing_level_swept",
    )
    missing = [k for k in required_keys if k not in signal]
    if missing:
        raise ValueError(f"Signal missing required keys: {missing}")


def compute_rr_from_levels(
    *,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
) -> float:
    if direction == "LONG":
        stop_dist = entry_price - stop_loss
        target_dist = target_price - entry_price
    elif direction == "SHORT":
        stop_dist = stop_loss - entry_price
        target_dist = entry_price - target_price
    else:
        raise ValueError("direction must be 'LONG' or 'SHORT'")

    if stop_dist <= 0:
        raise ValueError("Invalid stop distance (must be > 0).")
    if target_dist <= 0:
        raise ValueError("Invalid target distance (must be > 0).")
    return target_dist / stop_dist


def round_down_to_step(value: float, step: float) -> float:
    """
    Round down to the nearest multiple of `step`.

    Example: step=0.001, value=1.23456 => 1.234
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    return (value // step) * step


def compute_quantity_linear(
    *,
    entry_price: float,
    stop_loss: float,
    risk_amount: float,
    min_qty: Optional[float] = None,
    step_size: Optional[float] = None,
) -> float:
    """
    Compute base-asset quantity for linear futures such that:
      abs(PnL at stop) == risk_amount
    """
    stop_dist = abs(entry_price - stop_loss)
    if stop_dist <= 0:
        raise ValueError("stop_dist must be > 0")
    qty = risk_amount / stop_dist

    if step_size is not None:
        qty = round_down_to_step(qty, float(step_size))

    if min_qty is not None and qty < min_qty:
        return float(min_qty)

    return float(qty)


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    leverage: float = 10.0
    min_rr: float = 2.0

    # Optional execution constraints (exchange-specific). If you don't have these,
    # leave them as None.
    min_qty: Optional[float] = None
    step_size: Optional[float] = None


def plan_trade(
    *,
    signal: dict[str, Any],
    equity: float,
    risk_config: RiskConfig = RiskConfig(),
) -> dict[str, Any]:
    """
    Produce an exchange-agnostic trade plan dict:
    - side (BUY/SELL)
    - quantity (base asset units)
    - entry_price, stop_loss, target_price
    - risk_amount (in quote terms, e.g. USDC)
    - rr_ratio (revalidated)

    Note: leverage is not required for sizing under the linear PnL assumption,
    because risk is computed directly in price-difference space.
    Leverage affects margin usage, which is typically handled by the execution layer.
    """
    validate_signal(signal)
    direction = str(signal["direction"])
    entry_price = float(signal["entry_price"])
    stop_loss = float(signal["stop_loss"])
    target_price = float(signal["target_price"])

    rr = compute_rr_from_levels(
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
    )
    if rr < float(risk_config.min_rr):
        raise ValueError(f"RR too low: rr={rr:.4f} < min_rr={risk_config.min_rr}")

    risk_amount = float(equity) * float(risk_config.risk_per_trade)
    qty = compute_quantity_linear(
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_amount=risk_amount,
        min_qty=risk_config.min_qty,
        step_size=risk_config.step_size,
    )

    if qty <= 0:
        raise ValueError("Computed quantity must be > 0.")

    return {
        "direction": direction,
        "side": _direction_to_side(direction),
        "quantity": qty,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "rr_ratio": float(rr),
        "risk_amount": risk_amount,
        "equity": float(equity),
        "risk_per_trade": float(risk_config.risk_per_trade),
        "leverage": float(risk_config.leverage),
        "signal_index": signal.get("signal_index"),
        "metadata": {
            "htf_bias": signal.get("htf_bias"),
            "candle_body_ratio": signal.get("candle_body_ratio"),
            "swing_level_swept": signal.get("swing_level_swept"),
            "target_swing_level": signal.get("target_swing_level"),
        },
    }

