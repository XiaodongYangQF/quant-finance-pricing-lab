"""
Scenario analysis tools for Black-Scholes option pricing.

This module builds scenario tables for:

1. Spot and volatility shocks
2. Greek sensitivity across spot prices
3. Time decay analysis

The functions are designed to work with the package's current naming style:
BlackScholesInputs, black_scholes_price, and black_scholes_greeks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from quant_finance_pricing_lab.black_scholes import (
    BlackScholesInputs,
    black_scholes_price,
)

from quant_finance_pricing_lab.greeks import black_scholes_greeks


def _read_result_value(result, possible_names: list[str]) -> float:
    """
    Read a numeric value from either a dataclass result, dict result,
    or object with attributes.

    This makes the scenario module robust to small naming differences such as
    price, option_price, or value.
    """

    if isinstance(result, dict):
        for name in possible_names:
            if name in result:
                return float(result[name])

    for name in possible_names:
        if hasattr(result, name):
            return float(getattr(result, name))

    raise AttributeError(f"Could not find any of these fields: {possible_names}")


def _price_value(inputs: BlackScholesInputs) -> float:
    """
    Read Black-Scholes price from black_scholes_price output.
    """

    result = black_scholes_price(inputs)

    if isinstance(result, (int, float)):
        return float(result)

    return _read_result_value(
        result,
        possible_names=["price", "option_price", "value"],
    )


def _evaluate_inputs(inputs: BlackScholesInputs) -> dict[str, float]:
    """
    Evaluate price and Greeks for one Black-Scholes input object.
    """

    greeks_result = black_scholes_greeks(inputs)

    row = {}

    try:
        row["price"] = _read_result_value(
            greeks_result,
            possible_names=["price", "option_price", "value"],
        )
    except AttributeError:
        row["price"] = _price_value(inputs)

    greek_fields = ["delta", "gamma", "vega", "theta", "rho"]

    for field in greek_fields:
        try:
            row[field] = _read_result_value(greeks_result, [field])
        except AttributeError:
            row[field] = np.nan

    return row


def make_shock_grid(
    lower_pct: float,
    upper_pct: float,
    step_pct: float,
) -> list[float]:
    """
    Create a grid of percentage shocks in decimal form.

    Example
    -------
    make_shock_grid(-20, 20, 5)
    returns:
    [-0.20, -0.15, ..., 0.15, 0.20]
    """

    if step_pct <= 0:
        raise ValueError("step_pct must be positive.")

    if lower_pct > upper_pct:
        raise ValueError("lower_pct cannot be greater than upper_pct.")

    grid = np.arange(lower_pct, upper_pct + step_pct * 0.5, step_pct)

    return (grid / 100.0).round(10).tolist()


def build_spot_vol_scenario_table(
    base_inputs: BlackScholesInputs,
    spot_shocks: Iterable[float],
    vol_shocks: Iterable[float],
) -> pd.DataFrame:
    """
    Build a scenario table for spot and volatility shocks.

    Parameters
    ----------
    base_inputs:
        Base Black-Scholes inputs.
    spot_shocks:
        Spot shocks in decimal form. Example: [-0.10, 0.0, 0.10].
    vol_shocks:
        Volatility shocks in decimal form. Example: [-0.05, 0.0, 0.05].

    Returns
    -------
    pd.DataFrame
        Scenario table with price, Greeks, and changes from base case.
    """

    base_eval = _evaluate_inputs(base_inputs)
    rows = []

    for spot_shock in spot_shocks:
        for vol_shock in vol_shocks:
            scenario_spot = base_inputs.spot * (1.0 + spot_shock)
            scenario_vol = max(base_inputs.volatility + vol_shock, 1e-6)

            scenario_inputs = replace(
                base_inputs,
                spot=scenario_spot,
                volatility=scenario_vol,
            )

            scenario_eval = _evaluate_inputs(scenario_inputs)

            row = {
                "spot_shock": spot_shock,
                "spot_shock_pct": spot_shock * 100.0,
                "vol_shock": vol_shock,
                "vol_shock_pct_points": vol_shock * 100.0,
                "spot": scenario_spot,
                "volatility": scenario_vol,
                "volatility_pct": scenario_vol * 100.0,
            }

            for metric in ["price", "delta", "gamma", "vega", "theta", "rho"]:
                value = scenario_eval[metric]
                base_value = base_eval[metric]

                row[metric] = value
                row[f"{metric}_change"] = value - base_value

                if base_value != 0 and np.isfinite(base_value):
                    row[f"{metric}_change_pct"] = (value / base_value - 1.0) * 100.0
                else:
                    row[f"{metric}_change_pct"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_greek_sensitivity_table(
    base_inputs: BlackScholesInputs,
    spot_grid: Iterable[float],
) -> pd.DataFrame:
    """
    Build price and Greek sensitivity table across spot prices.
    """

    rows = []

    for spot in spot_grid:
        scenario_inputs = replace(base_inputs, spot=float(spot))
        scenario_eval = _evaluate_inputs(scenario_inputs)

        rows.append(
            {
                "spot": float(spot),
                "moneyness_S_over_K": float(spot) / base_inputs.strike,
                **scenario_eval,
            }
        )

    return pd.DataFrame(rows)


def build_time_decay_table(
    base_inputs: BlackScholesInputs,
    maturity_grid: Iterable[float],
) -> pd.DataFrame:
    """
    Build time decay table by changing time to maturity.

    Parameters
    ----------
    base_inputs:
        Base Black-Scholes inputs.
    maturity_grid:
        Grid of maturities in years.

    Returns
    -------
    pd.DataFrame
        Price and Greeks as maturity decreases.
    """

    rows = []

    for maturity in maturity_grid:
        maturity = max(float(maturity), 1e-6)

        scenario_inputs = replace(base_inputs, maturity=maturity)
        scenario_eval = _evaluate_inputs(scenario_inputs)

        rows.append(
            {
                "maturity": maturity,
                "days_to_maturity": maturity * 365.0,
                **scenario_eval,
            }
        )

    return pd.DataFrame(rows)