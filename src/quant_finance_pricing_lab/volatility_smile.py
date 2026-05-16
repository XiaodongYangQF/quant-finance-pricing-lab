"""
Volatility smile tools.

This module converts a cross-section of option market prices into
Black-Scholes implied volatilities by strike.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal

import numpy as np
import pandas as pd

from quant_finance_pricing_lab.implied_volatility import (
    implied_volatility,
    option_price_bounds,
)

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SmileInputs:
    """
    Market and contract inputs for a volatility smile calculation.
    """

    spot: float
    maturity: float
    rate: float
    dividend_yield: float = 0.0
    option_type: OptionType = "call"

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive.")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'.")


def build_volatility_smile(
    market_data: pd.DataFrame,
    smile_inputs: SmileInputs,
    strike_col: str = "strike",
    price_col: str = "market_price",
) -> pd.DataFrame:
    """
    Build an implied volatility smile table from option market prices.

    Parameters
    ----------
    market_data:
        DataFrame containing at least strike and market price columns.
    smile_inputs:
        Spot, maturity, rate, dividend yield, and option type.
    strike_col:
        Name of the strike column.
    price_col:
        Name of the market option price column.

    Returns
    -------
    pd.DataFrame
        Table with strike, market price, implied volatility, moneyness,
        forward moneyness, no-arbitrage bounds, and status.
    """

    smile_inputs.validate()

    required_cols = {strike_col, price_col}
    missing_cols = required_cols.difference(market_data.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    forward = smile_inputs.spot * exp(
        (smile_inputs.rate - smile_inputs.dividend_yield)
        * smile_inputs.maturity
    )

    rows = []

    for row_id, row in market_data.iterrows():
        strike = row[strike_col]
        market_price = row[price_col]

        try:
            strike = float(strike)
            market_price = float(market_price)

            if not np.isfinite(strike) or strike <= 0:
                raise ValueError("Invalid strike.")

            if not np.isfinite(market_price) or market_price <= 0:
                raise ValueError("Invalid market price.")

            lower_bound, upper_bound = option_price_bounds(
                spot=smile_inputs.spot,
                strike=strike,
                maturity=smile_inputs.maturity,
                rate=smile_inputs.rate,
                dividend_yield=smile_inputs.dividend_yield,
                option_type=smile_inputs.option_type,
            )

            iv_result = implied_volatility(
                market_price=market_price,
                spot=smile_inputs.spot,
                strike=strike,
                maturity=smile_inputs.maturity,
                rate=smile_inputs.rate,
                dividend_yield=smile_inputs.dividend_yield,
                option_type=smile_inputs.option_type,
            )

            implied_vol = iv_result.implied_volatility
            pricing_error = iv_result.pricing_error
            status = "ok"

        except Exception as error:
            lower_bound = np.nan
            upper_bound = np.nan
            implied_vol = np.nan
            pricing_error = np.nan
            status = str(error)

        rows.append(
            {
                "row_id": row_id,
                "strike": strike,
                "market_price": market_price,
                "implied_volatility": implied_vol,
                "implied_volatility_pct": (
                    implied_vol * 100 if np.isfinite(implied_vol) else np.nan
                ),
                "moneyness_K_over_S": (
                    strike / smile_inputs.spot
                    if np.isfinite(strike)
                    else np.nan
                ),
                "log_moneyness": (
                    log(strike / smile_inputs.spot)
                    if np.isfinite(strike) and strike > 0
                    else np.nan
                ),
                "forward": forward,
                "forward_moneyness_K_over_F": (
                    strike / forward
                    if np.isfinite(strike) and forward > 0
                    else np.nan
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "pricing_error": pricing_error,
                "status": status,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("strike").reset_index(drop=True)

    return result


def valid_smile_points(smile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows with successfully solved implied volatility.
    """

    return smile_df[
        smile_df["implied_volatility"].notna()
        & np.isfinite(smile_df["implied_volatility"])
        & (smile_df["status"] == "ok")
    ].copy()