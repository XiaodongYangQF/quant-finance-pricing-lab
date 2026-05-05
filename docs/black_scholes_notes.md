# Black-Scholes Notes

## Setup

Assume the stock price follows geometric Brownian motion under the risk-neutral measure:

\[
dS_t = r S_t dt + \sigma S_t dW_t^Q.
\]

For a European option with payoff \(\Phi(S_T)\), the no-arbitrage price is the discounted risk-neutral expectation:

\[
V_0 = e^{-rT} \mathbb{E}^Q[\Phi(S_T)].
\]

## European call and put

For a call option:

\[
C = S_0 N(d_1) - K e^{-rT}N(d_2).
\]

For a put option:

\[
P = K e^{-rT}N(-d_2) - S_0N(-d_1).
\]

where

\[
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
 d_2 = d_1 - \sigma\sqrt{T}.
\]

## Put-call parity

For European options without dividends:

\[
C - P = S_0 - K e^{-rT}.
\]

This identity is useful for checking implementation errors and market data quality.

## Portfolio interpretation

This notebook-style repo is not only about formulas. It shows that you can:

1. Translate financial theory into clean Python code.
2. Validate models using tests and known benchmark values.
3. Compare analytical pricing, tree pricing, Monte Carlo pricing, and PDE methods.
4. Explain derivatives clearly to both academic and industry audiences.
