"""Finite-difference pricing for the Black-Scholes PDE."""

from __future__ import annotations

import math

import numpy as np

from .black_scholes import OptionType


def implicit_fd_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: OptionType = "call",
    s_max: float | None = None,
    m_space: int = 200,
    n_time: int = 200,
) -> float:
    """Price a European option using an implicit finite-difference scheme.

    Parameters
    ----------
    m_space:
        Number of stock-price grid intervals.
    n_time:
        Number of time grid intervals.

    Notes
    -----
    This implementation is intentionally clear rather than optimised. It solves
    a dense linear system at each time step. A production implementation would
    normally use a tridiagonal solver.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    if m_space < 3 or n_time < 1:
        raise ValueError("m_space must be at least 3 and n_time at least 1.")

    if s_max is None:
        s_max = max(4.0 * K, 2.0 * S)
    if S >= s_max:
        raise ValueError("s_max must be greater than S.")

    dS = s_max / m_space
    dt = T / n_time
    grid_S = np.linspace(0.0, s_max, m_space + 1)

    if option_type == "call":
        values = np.maximum(grid_S - K, 0.0)
    else:
        values = np.maximum(K - grid_S, 0.0)

    i = np.arange(1, m_space)
    lower = -0.5 * dt * (sigma * sigma * i * i - r * i)
    diag = 1.0 + dt * (sigma * sigma * i * i + r)
    upper = -0.5 * dt * (sigma * sigma * i * i + r * i)

    A = np.zeros((m_space - 1, m_space - 1))
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], lower[1:])
    np.fill_diagonal(A[:, 1:], upper[:-1])

    for step in range(n_time - 1, -1, -1):
        tau = T - step * dt
        rhs = values[1:m_space].copy()

        if option_type == "call":
            boundary_low = 0.0
            boundary_high = s_max - K * math.exp(-r * tau)
        else:
            boundary_low = K * math.exp(-r * tau)
            boundary_high = 0.0

        rhs[0] -= lower[0] * boundary_low
        rhs[-1] -= upper[-1] * boundary_high
        values[0] = boundary_low
        values[m_space] = boundary_high
        values[1:m_space] = np.linalg.solve(A, rhs)

    return float(np.interp(S, grid_S, values))
