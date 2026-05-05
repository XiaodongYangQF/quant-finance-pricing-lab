"""Monte Carlo option pricing under geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .black_scholes import OptionType


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for Monte Carlo pricing output."""

    price: float
    standard_error: float
    n_paths: int
    antithetic: bool


def simulate_gbm_terminal(
    S: float,
    r: float,
    T: float,
    sigma: float,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate terminal stock prices under geometric Brownian motion."""
    if S <= 0:
        raise ValueError("S must be positive.")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma cannot be negative.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def price_european_monte_carlo(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Price a European option by Monte Carlo simulation.

    The standard error is computed from discounted simulated payoffs.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    terminal = simulate_gbm_terminal(S, r, T, sigma, n_paths, seed=seed, antithetic=antithetic)
    if option_type == "call":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0
    return MonteCarloResult(price=price, standard_error=standard_error, n_paths=n_paths, antithetic=antithetic)
