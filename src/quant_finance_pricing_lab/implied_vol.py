"""Implied volatility solver."""

from __future__ import annotations

import math

from .black_scholes import OptionType, black_scholes_price


def _no_arbitrage_bounds(S: float, K: float, r: float, T: float, option_type: OptionType) -> tuple[float, float]:
    discount = math.exp(-r * T)
    if option_type == "call":
        return max(S - K * discount, 0.0), S
    return max(K * discount - S, 0.0), K * discount


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: OptionType = "call",
    lower: float = 1e-8,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Compute Black-Scholes implied volatility using bisection.

    Parameters
    ----------
    market_price:
        Observed option price.
    lower, upper:
        Lower and upper volatility brackets. The default upper bound of 500%
        is intentionally wide for robustness in educational examples.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if market_price <= 0:
        raise ValueError("market_price must be positive.")
    if T <= 0:
        raise ValueError("Implied volatility requires positive T.")

    lb, ub = _no_arbitrage_bounds(S, K, r, T, option_type)
    eps = 1e-10
    if not (lb - eps <= market_price <= ub + eps):
        raise ValueError(
            f"market_price={market_price} violates no-arbitrage bounds [{lb}, {ub}] for a {option_type}."
        )

    low_price = black_scholes_price(S, K, r, T, lower, option_type)
    high_price = black_scholes_price(S, K, r, T, upper, option_type)

    if market_price < low_price - eps or market_price > high_price + eps:
        raise ValueError("The market price is not bracketed by the volatility interval.")

    low, high = lower, upper
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        price = black_scholes_price(S, K, r, T, mid, option_type)
        error = price - market_price
        if abs(error) < tolerance:
            return mid
        if error > 0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)
