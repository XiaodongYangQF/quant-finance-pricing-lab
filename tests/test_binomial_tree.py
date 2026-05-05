from quant_finance_pricing_lab.black_scholes import black_scholes_price
from quant_finance_pricing_lab.binomial_tree import crr_binomial_price


def test_crr_converges_to_black_scholes_call():
    S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20
    bs = black_scholes_price(S, K, r, T, sigma, "call")
    crr = crr_binomial_price(S, K, r, T, sigma, steps=1000, option_type="call")

    assert abs(crr - bs) < 0.02
