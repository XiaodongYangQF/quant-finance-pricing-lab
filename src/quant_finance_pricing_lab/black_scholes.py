from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm
import numpy as np


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesInputs:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = "call"


@dataclass(frozen=True)
class BlackScholesPriceResult:
    price: float
    d1: float
    d2: float


def validate_inputs(inputs: BlackScholesInputs) -> None:
    if inputs.spot <= 0:
        raise ValueError("Spot price must be positive.")

    if inputs.strike <= 0:
        raise ValueError("Strike price must be positive.")

    if inputs.maturity <= 0:
        raise ValueError("Time to maturity must be positive.")

    if inputs.volatility <= 0:
        raise ValueError("Volatility must be positive.")

    if inputs.option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")


def compute_d1_d2(inputs: BlackScholesInputs) -> tuple[float, float]:
    validate_inputs(inputs)

    s = inputs.spot
    k = inputs.strike
    t = inputs.maturity
    r = inputs.rate
    q = inputs.dividend_yield
    sigma = inputs.volatility

    d1 = (
        np.log(s / k) + (r - q + 0.5 * sigma**2) * t
    ) / (sigma * np.sqrt(t))

    d2 = d1 - sigma * np.sqrt(t)

    return float(d1), float(d2)


def black_scholes_price(inputs: BlackScholesInputs) -> BlackScholesPriceResult:
    validate_inputs(inputs)

    s = inputs.spot
    k = inputs.strike
    t = inputs.maturity
    r = inputs.rate
    q = inputs.dividend_yield
    option_type = inputs.option_type

    d1, d2 = compute_d1_d2(inputs)

    discount_r = np.exp(-r * t)
    discount_q = np.exp(-q * t)

    if option_type == "call":
        price = s * discount_q * norm.cdf(d1) - k * discount_r * norm.cdf(d2)
    else:
        price = k * discount_r * norm.cdf(-d2) - s * discount_q * norm.cdf(-d1)

    return BlackScholesPriceResult(
        price=float(price),
        d1=float(d1),
        d2=float(d2),
    )


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Put-call parity:

        C - P = S exp(-qT) - K exp(-rT)

    Returns the pricing gap:

        observed_left_side - theoretical_right_side
    """

    left_side = call_price - put_price
    right_side = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(
        -rate * maturity
    )

    return float(left_side - right_side)