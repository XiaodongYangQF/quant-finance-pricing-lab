"""Cox-Ross-Rubinstein binomial tree pricing."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .black_scholes import OptionType

ExerciseType = Literal["european", "american"]


def crr_binomial_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    steps: int = 200,
    option_type: OptionType = "call",
    exercise: ExerciseType = "european",
) -> float:
    """Price an option using the Cox-Ross-Rubinstein binomial tree.

    The function supports European and American exercise styles.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma cannot be negative.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'.")

    if T == 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / steps
    if sigma == 0:
        forward_discounted = S - K * math.exp(-r * T)
        return max(forward_discounted, 0.0) if option_type == "call" else max(-forward_discounted, 0.0)

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("Risk-neutral probability is outside [0, 1]. Try more steps or different parameters.")

    j = np.arange(steps + 1)
    stock = S * (u ** j) * (d ** (steps - j))
    if option_type == "call":
        values = np.maximum(stock - K, 0.0)
    else:
        values = np.maximum(K - stock, 0.0)

    discount = math.exp(-r * dt)
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1 : step + 2] + (1.0 - p) * values[0 : step + 1])
        if exercise == "american":
            j = np.arange(step + 1)
            stock = S * (u ** j) * (d ** (step - j))
            intrinsic = np.maximum(stock - K, 0.0) if option_type == "call" else np.maximum(K - stock, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
