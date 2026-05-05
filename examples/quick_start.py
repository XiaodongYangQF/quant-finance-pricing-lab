"""Quick-start example for quant-finance-pricing-lab."""

from quant_finance_pricing_lab import (
    black_scholes_price,
    bs_greeks,
    crr_binomial_price,
    implied_volatility,
    implicit_fd_price,
    price_european_monte_carlo,
)


def main() -> None:
    S = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.20

    call = black_scholes_price(S, K, r, T, sigma, "call")
    put = black_scholes_price(S, K, r, T, sigma, "put")
    greeks = bs_greeks(S, K, r, T, sigma, "call")
    iv = implied_volatility(call, S, K, r, T, "call")
    crr = crr_binomial_price(S, K, r, T, sigma, steps=500, option_type="call")
    mc = price_european_monte_carlo(S, K, r, T, sigma, "call", n_paths=100_000, seed=42)
    fd = implicit_fd_price(S, K, r, T, sigma, "call", m_space=120, n_time=120)

    print(f"Black-Scholes call: {call:.4f}")
    print(f"Black-Scholes put:  {put:.4f}")
    print(f"Call delta:         {greeks['delta']:.4f}")
    print(f"Implied vol:        {iv:.4f}")
    print(f"CRR call:           {crr:.4f}")
    print(f"MC call:            {mc.price:.4f} ± {1.96 * mc.standard_error:.4f}")
    print(f"FD call:            {fd:.4f}")


if __name__ == "__main__":
    main()
