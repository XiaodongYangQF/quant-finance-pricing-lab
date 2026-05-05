"""Analytical Black-Scholes Greeks."""

from __future__ import annotations

import math
from typing import Dict

from .black_scholes import OptionType, d1_d2, norm_cdf, norm_pdf


def bs_greeks(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: OptionType = "call",
) -> Dict[str, float]:
    """Return analytical Black-Scholes Greeks for a European option.

    The Greeks are reported in natural units:

    - delta: change in option value per unit change in spot
    - gamma: change in delta per unit change in spot
    - vega: change in option value per 1.00 change in volatility, not per 1%
    - theta: change in option value per year
    - rho: change in option value per 1.00 change in risk-free rate, not per 1%
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if T <= 0:
        raise ValueError("Greeks require positive time to maturity T.")
    if sigma <= 0:
        raise ValueError("Greeks require positive volatility sigma.")

    d1, d2 = d1_d2(S, K, r, T, sigma)
    pdf_d1 = norm_pdf(d1)
    discount = math.exp(-r * T)
    sqrt_t = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_t)
    vega = S * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = norm_cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_t) - r * K * discount * norm_cdf(d2)
        rho = K * T * discount * norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_t) + r * K * discount * norm_cdf(-d2)
        rho = -K * T * discount * norm_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
