from quant_finance_pricing_lab.black_scholes import black_scholes_price
from quant_finance_pricing_lab.monte_carlo import price_european_monte_carlo


def test_monte_carlo_price_within_confidence_band():
    S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20
    bs = black_scholes_price(S, K, r, T, sigma, "call")
    mc = price_european_monte_carlo(S, K, r, T, sigma, "call", n_paths=200_000, seed=7)

    assert abs(mc.price - bs) < 4.0 * mc.standard_error
