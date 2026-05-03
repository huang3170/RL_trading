from __future__ import annotations

from typing import Literal


StateRepresentation = Literal["agent", "full", "lctc"]
RewardType = Literal["pnl", "symmetric", "asymmetric"]
EncoderType = Literal["mlp", "transformer"]
