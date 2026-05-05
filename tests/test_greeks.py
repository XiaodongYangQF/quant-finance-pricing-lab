from quant_finance_pricing_lab.greeks import bs_greeks


def test_call_greeks_known_values():
    greeks = bs_greeks(100.0, 100.0, 0.05, 1.0, 0.20, "call")

    assert abs(greeks["delta"] - 0.6368) < 1e-4
    assert abs(greeks["gamma"] - 0.0188) < 1e-4
    assert abs(greeks["vega"] - 37.5240) < 1e-3
    assert greeks["theta"] < 0
    assert greeks["rho"] > 0
