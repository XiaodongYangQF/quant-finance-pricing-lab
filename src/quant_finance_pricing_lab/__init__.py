"""Quant Finance Pricing Lab.

Educational implementations for derivatives pricing and numerical methods.
"""

from .black_scholes import black_scholes_price, put_call_parity_gap
from .greeks import bs_greeks
from .implied_vol import implied_volatility
from .binomial_tree import crr_binomial_price
from .monte_carlo import MonteCarloResult, price_european_monte_carlo, simulate_gbm_terminal
from .finite_difference import implicit_fd_price

__all__ = [
    "black_scholes_price",
    "put_call_parity_gap",
    "bs_greeks",
    "implied_volatility",
    "crr_binomial_price",
    "MonteCarloResult",
    "price_european_monte_carlo",
    "simulate_gbm_terminal",
    "implicit_fd_price",
]
