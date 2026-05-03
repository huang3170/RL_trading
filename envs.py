from __future__ import annotations

from typing import Optional, Tuple
import math
import random

import numpy as np

from config import EnvConfig
from market_data import EventStream, MarketEvent
from rewards import compute_reward
from state import ObservationStacker, StateBuilder


class EventDrivenMarketMakingEnv:
    """
    Event-driven env.

    One call to step(action) does:
    1. Decode action into ask/bid quotes and optional market-order fraction.
    2. Consume exactly one historical market event.
    3. Simulate fills caused by the event.
    4. Update inventory.
    5. Compute event-to-event reward.
    6. Return next event-state.
    """

    def __init__(self, stream: EventStream, cfg: EnvConfig):
        self.stream = stream
        self.cfg = cfg
        self.state_builder = StateBuilder(cfg)
        self.stacker = ObservationStacker(cfg.window_size, cfg.state_representation)
        self.inventory = 0
        self.theta_ask = 1.0
        self.theta_bid = 1.0
        self.active_ask_price: Optional[float] = None
        self.active_bid_price: Optional[float] = None
        self.done = False

    def reset(self):
        event = self.stream.reset()
        self.state_builder.reset(event)
        self.stacker.reset()
        self.inventory = 0
        self.theta_ask = 1.0
        self.theta_bid = 1.0
        self.active_ask_price = None
        self.active_bid_price = None
        self.done = False
        obs = self.state_builder.build(event, self.inventory, self.theta_ask, self.theta_bid)
        return self.stacker.push(obs)

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        prev_event = self.stream.current()
        prev_mid = prev_event.mid

        self.theta_ask, self.theta_bid, mo_frac = self._decode_action(action)
        self.active_ask_price, self.active_bid_price = self._quotes_from_action(prev_event)

        # Optional inventory-clearing market order.
        market_order_size = self._market_order_size(mo_frac)
        self.inventory += market_order_size

        # Consume exactly ONE market event: this is the event-driven transition.
        event, self.done = self.stream.next()

        matched_ask, matched_bid = self._simulate_fills(prev_event, event)

        # ask fill means we sell -> inventory decreases
        # bid fill means we buy -> inventory increases
        self.inventory -= int(matched_ask)
        self.inventory += int(matched_bid)

        delta_mid = event.mid - prev_mid
        reward_dict = compute_reward(
            reward_type=self.cfg.reward_type,
            matched_ask=matched_ask,
            matched_bid=matched_bid,
            ask_price=self.active_ask_price,
            bid_price=self.active_bid_price,
            mid_price=event.mid,
            inventory=self.inventory,
            delta_mid=delta_mid,
            eta=self.cfg.eta,
            inventory_penalty=self.cfg.inventory_penalty,
        )

        # Enforce inventory constraints by penalizing and clipping.
        if self.inventory < self.cfg.min_inventory or self.inventory > self.cfg.max_inventory:
            reward_dict["reward"] -= abs(self.inventory) * 1e-3
            self.inventory = int(np.clip(self.inventory, self.cfg.min_inventory, self.cfg.max_inventory))

        obs_raw = self.state_builder.build(event, self.inventory, self.theta_ask, self.theta_bid)
        obs = self.stacker.push(obs_raw)
        self.state_builder.update_market_memory(event)

        info = {
            **reward_dict,
            "timestamp": event.timestamp,
            "mid": event.mid,
            "spread": event.spread,
            "inventory": self.inventory,
            "theta_ask": self.theta_ask,
            "theta_bid": self.theta_bid,
            "ask_price": self.active_ask_price,
            "bid_price": self.active_bid_price,
            "matched_ask": matched_ask,
            "matched_bid": matched_bid,
            "market_order_size": market_order_size,
        }
        return obs, float(reward_dict["reward"]), self.done, info

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        theta_ask = (action[0] + 1.0) * 0.5 * self.cfg.max_theta
        theta_bid = (action[1] + 1.0) * 0.5 * self.cfg.max_theta
        mo_frac = (action[2] + 1.0) * 0.5 * self.cfg.max_market_order_fraction
        return float(theta_ask), float(theta_bid), float(mo_frac)

    def _quotes_from_action(self, event: MarketEvent) -> Tuple[float, float]:
        spread_scale = max(self.cfg.tick_size, event.spread / 2.0)
        ask = event.mid + self.theta_ask * spread_scale
        bid = event.mid - self.theta_bid * spread_scale
        return self._round_to_tick(ask), self._round_to_tick(bid)

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.cfg.tick_size) * self.cfg.tick_size

    def _market_order_size(self, mo_frac: float) -> int:
        if self.inventory == 0:
            return 0
        raw = -mo_frac * self.inventory
        size = int(round(raw / self.cfg.lot_size) * self.cfg.lot_size)
        return size

    def _simulate_fills(self, prev_event: MarketEvent, event: MarketEvent) -> Tuple[int, int]:
        """
        Simplified event-driven fill logic.

        Replace this with queue-position tracking if your data has enough fields.
        Logic:
        - ask quote fills if market best bid moves up through our ask, or buy pressure is large
        - bid quote fills if market best ask moves down through our bid, or sell pressure is large
        """
        matched_ask = 0
        matched_bid = 0

        if self.active_ask_price is not None:
            if event.best_bid >= self.active_ask_price or event.signed_trade_volume > 0:
                # Fill probability decreases when quote is deeper from mid.
                dist = max(self.active_ask_price - prev_event.mid, 0.0)
                p = math.exp(-dist / max(prev_event.spread, self.cfg.tick_size))
                if random.random() < min(1.0, p):
                    matched_ask = self.cfg.order_size

        if self.active_bid_price is not None:
            if event.best_ask <= self.active_bid_price or event.signed_trade_volume < 0:
                dist = max(prev_event.mid - self.active_bid_price, 0.0)
                p = math.exp(-dist / max(prev_event.spread, self.cfg.tick_size))
                if random.random() < min(1.0, p):
                    matched_bid = self.cfg.order_size

        return matched_ask, matched_bid
