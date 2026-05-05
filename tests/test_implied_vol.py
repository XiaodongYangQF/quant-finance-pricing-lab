from quant_finance_pricing_lab.black_scholes import black_scholes_price
from quant_finance_pricing_lab.implied_vol import implied_volatility


def test_implied_volatility_recovers_true_sigma():
    S, K, r, T, sigma = 100.0, 105.0, 0.04, 0.8, 0.30
    price = black_scholes_price(S, K, r, T, sigma, "call")
    iv = implied_volatility(price, S, K, r, T, "call")

    assert abs(iv - sigma) < 1e-6
