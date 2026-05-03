from __future__ import annotations

from typing import Dict

from trading_types import RewardType


def compute_reward(
    reward_type: RewardType,
    matched_ask: float,
    matched_bid: float,
    ask_price: float,
    bid_price: float,
    mid_price: float,
    inventory: float,
    delta_mid: float,
    eta: float,
    inventory_penalty: float = 0.0,
) -> Dict[str, float]:
    psi_a = matched_ask * (ask_price - mid_price)
    psi_b = matched_bid * (mid_price - bid_price)
    spread_capture = psi_a + psi_b
    inventory_pnl = inventory * delta_mid
    raw_pnl = spread_capture + inventory_pnl

    if reward_type == "pnl":
        reward = raw_pnl
    elif reward_type == "symmetric":
        reward = raw_pnl - eta * inventory_pnl
    elif reward_type == "asymmetric":
        reward = raw_pnl - max(0.0, eta * inventory_pnl)
    else:
        raise ValueError(reward_type)

    reward -= inventory_penalty * inventory * inventory

    return {
        "reward": float(reward),
        "raw_pnl": float(raw_pnl),
        "spread_capture": float(spread_capture),
        "inventory_pnl": float(inventory_pnl),
        "psi_a": float(psi_a),
        "psi_b": float(psi_b),
    }
