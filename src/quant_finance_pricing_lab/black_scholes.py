"""Black-Scholes pricing utilities.

The functions in this module avoid external dependencies beyond the Python
standard library. The normal CDF is computed using ``math.erf``.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

OptionType = Literal["call", "put"]


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("Spot price S must be positive.")
    if K <= 0:
        raise ValueError("Strike K must be positive.")
    if T < 0:
        raise ValueError("Time to maturity T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def d1_d2(S: float, K: float, r: float, T: float, sigma: float) -> Tuple[float, float]:
    """Return Black-Scholes d1 and d2.

    Parameters
    ----------
    S, K, r, T, sigma:
        Spot, strike, continuously compounded risk-free rate, time to maturity,
        and volatility.
    """
    _validate_inputs(S, K, T, sigma)
    if T == 0 or sigma == 0:
        raise ValueError("d1 and d2 are undefined when T=0 or sigma=0.")

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Price a European call or put option using the Black-Scholes formula.

    The implementation assumes no dividends. For immediate maturity or zero
    volatility, the function returns the discounted deterministic payoff.
    """
    _validate_inputs(S, K, T, sigma)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if T == 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if sigma == 0:
        forward_discounted = S - K * math.exp(-r * T)
        return max(forward_discounted, 0.0) if option_type == "call" else max(-forward_discounted, 0.0)

    d1, d2 = d1_d2(S, K, r, T, sigma)
    discount = math.exp(-r * T)

    if option_type == "call":
        return S * norm_cdf(d1) - K * discount * norm_cdf(d2)
    return K * discount * norm_cdf(-d2) - S * norm_cdf(-d1)


def put_call_parity_gap(call_price: float, put_price: float, S: float, K: float, r: float, T: float) -> float:
    """Return the put-call parity gap.

    For European options with no dividends:

    C - P = S - K exp(-rT)

    A value close to zero indicates parity consistency.
    """
    return call_price - put_price - (S - K * math.exp(-r * T))
