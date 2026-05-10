# Quant Finance Pricing Lab

A compact Python laboratory for derivatives pricing, Greeks, implied volatility, tree methods, Monte Carlo simulation, and finite-difference pricing.

This repository is designed as a clean GitHub portfolio project for quantitative finance, derivatives, risk, model validation, and quant analyst roles. It focuses on transparent implementations rather than black-box libraries.

## Why this repo exists

This project demonstrates the core foundations expected in quantitative finance:

- Black-Scholes pricing for European options
- Put-call parity checks
- Analytical Greeks
- Implied volatility inversion
- Cox-Ross-Rubinstein binomial tree pricing
- Monte Carlo option pricing with standard errors
- Finite-difference pricing for the Black-Scholes PDE

The code uses synthetic examples only. No proprietary data, unpublished research code, or OptionMetrics data are included.

## Repository structure

```text
quant-finance-pricing-lab/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   └── quant_finance_pricing_lab/
│       ├── __init__.py
│       ├── black_scholes.py
│       ├── greeks.py
│       ├── implied_vol.py
│       ├── binomial_tree.py
│       ├── monte_carlo.py
│       └── finite_difference.py
│
├── tests/
│   ├── test_black_scholes.py
│   ├── test_greeks.py
│   ├── test_implied_vol.py
│   ├── test_binomial_tree.py
│   └── test_monte_carlo.py
│
├── examples/
│   └── quick_start.py
│
├── notebooks/
│   ├── 01_black_scholes_derivation.ipynb
│   ├── 02_greeks_and_hedging.ipynb
│   ├── 03_implied_volatility_solver.ipynb
│   ├── 04_binomial_tree_pricing.ipynb
│   ├── 05_monte_carlo_pricing.ipynb
│   └── 06_finite_difference_black_scholes.ipynb
│
└── docs/
    └── theory/
        └── black_scholes/
            ├── README.md
            └── black_scholes_three_derivations.pdf

## Installation

Clone the repository and install it locally:

```bash
git clone https://github.com/YOUR_USERNAME/quant-finance-pricing-lab.git
cd quant-finance-pricing-lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
```

## Quick example

```python
from quant_finance_pricing_lab import (
    black_scholes_price,
    bs_greeks,
    implied_volatility,
    crr_binomial_price,
    price_european_monte_carlo,
)

S = 100
K = 100
r = 0.05
T = 1.0
sigma = 0.20

call = black_scholes_price(S, K, r, T, sigma, option_type="call")
put = black_scholes_price(S, K, r, T, sigma, option_type="put")
greeks = bs_greeks(S, K, r, T, sigma, option_type="call")
iv = implied_volatility(call, S, K, r, T, option_type="call")
crr = crr_binomial_price(S, K, r, T, sigma, steps=500, option_type="call")
mc = price_european_monte_carlo(S, K, r, T, sigma, option_type="call", n_paths=100_000, seed=42)

print(f"Black-Scholes call: {call:.4f}")
print(f"Black-Scholes put:  {put:.4f}")
print(f"Call delta:         {greeks['delta']:.4f}")
print(f"Implied vol:        {iv:.4f}")
print(f"CRR call:           {crr:.4f}")
print(f"MC call:            {mc.price:.4f} ± {1.96 * mc.standard_error:.4f}")
```

## Methods included

### 1. Black-Scholes model

The Black-Scholes model prices European call and put options under lognormal dynamics:

\[
dS_t = r S_t dt + \sigma S_t dW_t.
\]

The implementation provides prices for calls and puts, together with put-call parity checks.

### 2. Greeks

The repo computes analytical Greeks:

- Delta
- Gamma
- Vega
- Theta
- Rho

These are useful for hedging intuition, model validation, and interview preparation.

### 3. Implied volatility

The implied volatility solver uses robust bisection. It avoids relying on external optimisation libraries, making the method easy to understand and debug.

### 4. Binomial tree

The Cox-Ross-Rubinstein model is implemented for European and American options. It is useful for understanding discrete-time risk-neutral valuation and early exercise.

### 5. Monte Carlo pricing

The Monte Carlo module simulates terminal stock prices under geometric Brownian motion and reports both the option value and standard error. Antithetic variates are available as a variance-reduction option.

### 6. Finite difference method

The finite-difference module prices European options by solving the Black-Scholes PDE on a stock-price grid using an implicit scheme.

## Suggested GitHub description

> Python implementations of derivatives pricing models: Black-Scholes, Greeks, implied volatility, binomial trees, Monte Carlo, and finite-difference methods.

## Suggested topics

```text
quantitative-finance, derivatives-pricing, black-scholes, greeks, implied-volatility, monte-carlo, binomial-tree, finite-difference, python
```

## Future extensions

- Dividend yield support
- Barrier options
- Asian options
- Heston/Bates simulation
- Calibration examples
- Local volatility demo
- Risk-neutral density extraction from option prices

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice and should not be used for live trading or production risk management without independent validation.
