import math

from quant_finance_pricing_lab.black_scholes import black_scholes_price, put_call_parity_gap


def test_black_scholes_prices_known_values():
    S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20
    call = black_scholes_price(S, K, r, T, sigma, "call")
    put = black_scholes_price(S, K, r, T, sigma, "put")

    assert abs(call - 10.4506) < 1e-4
    assert abs(put - 5.5735) < 1e-4


def test_put_call_parity_gap_is_zero():
    S, K, r, T, sigma = 100.0, 95.0, 0.03, 0.75, 0.25
    call = black_scholes_price(S, K, r, T, sigma, "call")
    put = black_scholes_price(S, K, r, T, sigma, "put")

    assert math.isclose(put_call_parity_gap(call, put, S, K, r, T), 0.0, abs_tol=1e-10)
