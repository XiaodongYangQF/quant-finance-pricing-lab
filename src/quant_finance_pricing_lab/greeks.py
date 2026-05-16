from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm
import numpy as np

from quant_finance_pricing_lab.black_scholes import (
    BlackScholesInputs,
    black_scholes_price,
    compute_d1_d2,
    validate_inputs,
)


@dataclass(frozen=True)
class BlackScholesGreeksResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    d1: float
    d2: float


def black_scholes_greeks(inputs: BlackScholesInputs) -> BlackScholesGreeksResult:
    validate_inputs(inputs)

    s = inputs.spot
    k = inputs.strike
    t = inputs.maturity
    r = inputs.rate
    q = inputs.dividend_yield
    sigma = inputs.volatility
    option_type = inputs.option_type

    price_result = black_scholes_price(inputs)
    d1, d2 = compute_d1_d2(inputs)

    discount_r = np.exp(-r * t)
    discount_q = np.exp(-q * t)

    pdf_d1 = norm.pdf(d1)

    gamma = discount_q * pdf_d1 / (s * sigma * np.sqrt(t))
    vega_raw = s * discount_q * pdf_d1 * np.sqrt(t)

    if option_type == "call":
        delta = discount_q * norm.cdf(d1)

        theta_annual = (
            -s * discount_q * pdf_d1 * sigma / (2 * np.sqrt(t))
            - r * k * discount_r * norm.cdf(d2)
            + q * s * discount_q * norm.cdf(d1)
        )

        rho_raw = k * t * discount_r * norm.cdf(d2)

    else:
        delta = discount_q * (norm.cdf(d1) - 1)

        theta_annual = (
            -s * discount_q * pdf_d1 * sigma / (2 * np.sqrt(t))
            + r * k * discount_r * norm.cdf(-d2)
            - q * s * discount_q * norm.cdf(-d1)
        )

        rho_raw = -k * t * discount_r * norm.cdf(-d2)

    return BlackScholesGreeksResult(
        price=float(price_result.price),
        delta=float(delta),
        gamma=float(gamma),

        # Dashboard convention:
        # Vega = price change for 1 percentage point volatility change
        vega=float(vega_raw / 100),

        # Theta = price change per calendar day
        theta=float(theta_annual / 365),

        # Rho = price change for 1 percentage point interest-rate change
        rho=float(rho_raw / 100),

        d1=float(d1),
        d2=float(d2),
    )