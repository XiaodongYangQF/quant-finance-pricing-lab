from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from math import exp

from scipy.optimize import brentq

from quant_finance_pricing_lab.black_scholes import (
    BlackScholesInputs,
    black_scholes_price,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    implied_volatility: float
    market_price: float
    model_price: float
    pricing_error: float
    converged: bool


def _intrinsic_lower_bound(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> float:
    import numpy as np

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)

    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0)

    return max(discounted_strike - discounted_spot, 0.0)


def option_price_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> tuple[float, float]:
    """
    Compute no-arbitrage lower and upper bounds for European options
    under continuous dividend yield.
    """

    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    if option_type == "call":
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        upper_bound = discounted_spot
    else:
        lower_bound = max(discounted_strike - discounted_spot, 0.0)
        upper_bound = discounted_strike

    return lower_bound, upper_bound




def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    lower_vol: float = 1e-6,
    upper_vol: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    if spot <= 0:
        raise ValueError("Spot price must be positive.")

    if strike <= 0:
        raise ValueError("Strike price must be positive.")

    if maturity <= 0:
        raise ValueError("Time to maturity must be positive.")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    lower_bound = _intrinsic_lower_bound(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    if market_price < lower_bound:
        raise ValueError(
            f"Market price {market_price:.6f} is below the no-arbitrage lower bound "
            f"{lower_bound:.6f}."
        )

    def objective(volatility: float) -> float:
        inputs = BlackScholesInputs(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )

        return black_scholes_price(inputs).price - market_price

    iv = brentq(
        objective,
        lower_vol,
        upper_vol,
        xtol=tolerance,
        maxiter=max_iterations,
    )

    final_inputs = BlackScholesInputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=iv,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    model_price = black_scholes_price(final_inputs).price

    return ImpliedVolatilityResult(
        implied_volatility=float(iv),
        market_price=float(market_price),
        model_price=float(model_price),
        pricing_error=float(model_price - market_price),
        converged=True,
    )