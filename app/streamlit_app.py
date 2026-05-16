from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from quant_finance_pricing_lab.black_scholes import BlackScholesInputs
from quant_finance_pricing_lab.greeks import black_scholes_greeks
from quant_finance_pricing_lab.implied_volatility import implied_volatility
# from quant_finance_pricing_lab.scenario_analysis import (
#     build_greek_sensitivity_table,
#     build_spot_vol_scenario_table,
# )


from quant_finance_pricing_lab.scenario_analysis import (
    make_shock_grid,
    build_spot_vol_scenario_table,
    build_greek_sensitivity_table,
    build_time_decay_table,
)


from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Make src/ importable when running Streamlit from repo root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from quant_finance_pricing_lab.black_scholes import BlackScholesInputs
from quant_finance_pricing_lab.greeks import black_scholes_greeks

from quant_finance_pricing_lab.volatility_smile import (
    SmileInputs,
    build_volatility_smile,
    valid_smile_points,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Option Pricing & Greeks Dashboard",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# Helper functions
# ============================================================

def build_spot_grid(spot: float, strike: float, n: int = 121) -> np.ndarray:
    lower = max(0.01, min(spot, strike) * 0.5)
    upper = max(spot, strike) * 1.5
    return np.linspace(lower, upper, n)


def payoff_at_maturity(
    spot_at_maturity: np.ndarray,
    strike: float,
    option_type: str,
) -> np.ndarray:
    if option_type == "call":
        return np.maximum(spot_at_maturity - strike, 0.0)

    return np.maximum(strike - spot_at_maturity, 0.0)


def make_payoff_figure(
    inputs: BlackScholesInputs,
    option_price: float,
) -> go.Figure:
    spot_grid = build_spot_grid(inputs.spot, inputs.strike)

    gross_payoff = payoff_at_maturity(
        spot_grid,
        inputs.strike,
        inputs.option_type,
    )

    net_payoff = gross_payoff - option_price

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spot_grid,
            y=gross_payoff,
            mode="lines",
            name="Gross payoff",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spot_grid,
            y=net_payoff,
            mode="lines",
            name="Net payoff",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        annotation_text="Break-even line",
        annotation_position="bottom right",
    )

    fig.add_vline(
        x=inputs.strike,
        line_dash="dot",
        annotation_text="Strike",
        annotation_position="top",
    )

    fig.update_layout(
        height=430,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Underlying Price at Maturity",
        yaxis_title="Payoff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
    )

    return fig


def build_sensitivity_dataframe(inputs: BlackScholesInputs) -> pd.DataFrame:
    spot_grid = build_spot_grid(inputs.spot, inputs.strike)

    rows = []

    for spot_value in spot_grid:
        scenario_inputs = BlackScholesInputs(
            spot=float(spot_value),
            strike=inputs.strike,
            maturity=inputs.maturity,
            rate=inputs.rate,
            volatility=inputs.volatility,
            dividend_yield=inputs.dividend_yield,
            option_type=inputs.option_type,
        )

        result = black_scholes_greeks(scenario_inputs)

        rows.append(
            {
                "spot": spot_value,
                "price": result.price,
                "delta": result.delta,
                "gamma": result.gamma,
                "vega": result.vega,
                "theta": result.theta,
                "rho": result.rho,
            }
        )

    return pd.DataFrame(rows)


def make_single_sensitivity_figure(
    df: pd.DataFrame,
    y_col: str,
    y_title: str,
    current_spot: float,
    strike: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["spot"],
            y=df[y_col],
            mode="lines",
            name=y_title,
        )
    )

    fig.add_vline(
        x=current_spot,
        line_dash="dot",
        annotation_text="S",
        annotation_position="top",
    )

    fig.add_vline(
        x=strike,
        line_dash="dash",
        annotation_text="K",
        annotation_position="bottom",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Spot Price",
        yaxis_title=y_title,
        hovermode="x unified",
        showlegend=False,
    )

    return fig


def format_metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


# ============================================================
# App layout
# ============================================================

st.title("Option Pricing & Greeks Dashboard")

st.markdown(
    """
    This Streamlit dashboard extends the static website version with a stronger
    Python-based implementation of Black-Scholes pricing, Greeks, payoff profiles,
    and sensitivity analysis.
    """
)


# ============================================================
# Sidebar inputs
# ============================================================

st.sidebar.header("Model Inputs")

option_type = st.sidebar.selectbox(
    "Option Type",
    options=["call", "put"],
    index=0,
)

spot = st.sidebar.number_input(
    "Spot Price S",
    min_value=0.01,
    value=100.0,
    step=1.0,
)

strike = st.sidebar.number_input(
    "Strike Price K",
    min_value=0.01,
    value=100.0,
    step=1.0,
)

maturity = st.sidebar.number_input(
    "Time to Maturity T (years)",
    min_value=0.001,
    value=1.0,
    step=0.05,
)

rate = st.sidebar.number_input(
    "Risk-free Rate r",
    value=0.05,
    step=0.005,
    format="%.4f",
)

dividend_yield = st.sidebar.number_input(
    "Dividend Yield q",
    value=0.00,
    step=0.005,
    format="%.4f",
)

volatility = st.sidebar.number_input(
    "Volatility σ",
    min_value=0.001,
    value=0.20,
    step=0.01,
    format="%.4f",
)


inputs = BlackScholesInputs(
    spot=spot,
    strike=strike,
    maturity=maturity,
    rate=rate,
    volatility=volatility,
    dividend_yield=dividend_yield,
    option_type=option_type,
)

result = black_scholes_greeks(inputs)


# ============================================================
# Main results
# ============================================================

st.subheader("Pricing Results")

metric_cols = st.columns(6)

metric_cols[0].metric("Option Price", format_metric(result.price))
metric_cols[1].metric("Delta", format_metric(result.delta))
metric_cols[2].metric("Gamma", format_metric(result.gamma))
metric_cols[3].metric("Vega / 1% vol", format_metric(result.vega))
metric_cols[4].metric("Theta / day", format_metric(result.theta))
metric_cols[5].metric("Rho / 1% rate", format_metric(result.rho))


with st.expander("Model details"):
    detail_cols = st.columns(2)

    with detail_cols[0]:
        st.write("**Inputs**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": [
                        "Spot price",
                        "Strike price",
                        "Maturity",
                        "Risk-free rate",
                        "Dividend yield",
                        "Volatility",
                        "Option type",
                    ],
                    "Value": [
                        inputs.spot,
                        inputs.strike,
                        inputs.maturity,
                        inputs.rate,
                        inputs.dividend_yield,
                        inputs.volatility,
                        inputs.option_type,
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with detail_cols[1]:
        st.write("**Black-Scholes terms**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Term": ["d1", "d2"],
                    "Value": [result.d1, result.d2],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# Payoff chart
# ============================================================

st.subheader("Payoff Profile")

payoff_fig = make_payoff_figure(inputs, result.price)

st.plotly_chart(
    payoff_fig,
    use_container_width=True,
)


# ============================================================
# Sensitivity charts
# ============================================================

st.subheader("Greek Sensitivity Analysis")

st.markdown(
    """
    Each chart shows how the option value or Greek changes as the underlying
    spot price changes, while keeping the strike, maturity, volatility,
    risk-free rate, and dividend yield fixed.
    """
)

sensitivity_df = build_sensitivity_dataframe(inputs)

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "price",
            "Option Value",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )

with row1_col2:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "delta",
            "Delta",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )

with row1_col3:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "gamma",
            "Gamma",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "vega",
            "Vega / 1% vol",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )

with row2_col2:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "theta",
            "Theta / day",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )

with row2_col3:
    st.plotly_chart(
        make_single_sensitivity_figure(
            sensitivity_df,
            "rho",
            "Rho / 1% rate",
            inputs.spot,
            inputs.strike,
        ),
        use_container_width=True,
    )


# ============================================================
# Data table
# ============================================================

with st.expander("Show sensitivity data"):
    st.dataframe(
        sensitivity_df.round(6),
        use_container_width=True,
    )




# ============================================================
# Main pricing dashboard
# ============================================================

# Pricing results
# Model details
# Payoff chart
# Greek sensitivity analysis
# Sensitivity data table


# ============================================================
# Extended tools
# ============================================================

st.divider()

tab_iv, tab_smile, tab_scenario = st.tabs(
    [
        "Implied Volatility",
        "Volatility Smile",
        "Scenario Analysis",
    ]
)


# ============================================================
# Tab 1: Implied Volatility
# ============================================================

with tab_iv:
    st.subheader("Implied Volatility Calculator")

    st.markdown(
        """
        This tool solves for the Black-Scholes implied volatility from an observed
        market option price.
        """
    )

    iv_col1, iv_col2 = st.columns(2)

    with iv_col1:
        market_price = st.number_input(
            "Market Option Price",
            min_value=0.0001,
            value=float(result.price),
            step=0.1,
            format="%.4f",
            key="iv_market_price",
        )

    with iv_col2:
        st.write("Current model price")
        st.metric("Black-Scholes Price", format_metric(result.price))

    try:
        iv_result = implied_volatility(
            market_price=market_price,
            spot=inputs.spot,
            strike=inputs.strike,
            maturity=inputs.maturity,
            rate=inputs.rate,
            dividend_yield=inputs.dividend_yield,
            option_type=inputs.option_type,
        )

        iv_metric_col1, iv_metric_col2, iv_metric_col3 = st.columns(3)

        iv_metric_col1.metric(
            "Implied Volatility",
            f"{iv_result.implied_volatility:.4%}",
        )

        iv_metric_col2.metric(
            "Model Price at IV",
            f"{iv_result.model_price:.4f}",
        )

        iv_metric_col3.metric(
            "Pricing Error",
            f"{iv_result.pricing_error:.8f}",
        )

        st.info(
            "The implied volatility is the volatility input that makes the "
            "Black-Scholes model price equal to the observed market price."
        )

    except ValueError as error:
        st.error(str(error))


with tab_smile:
    st.subheader("Volatility Smile from Market Prices")

    st.markdown(
        """
        This tool solves Black-Scholes implied volatility across a cross-section
        of option market prices. You can paste CSV data or upload a CSV file.
        """
    )

    sample_data = """strike,market_price
80,22.1746
85,18.1600
90,14.6288
95,12.0000
100,10.4506
105,8.2000
110,6.0401
115,4.5000
120,3.2475
"""

    uploaded_file = st.file_uploader(
        "Upload CSV file with strike and market price columns",
        type=["csv"],
        key="smile_csv_upload",
    )

    if uploaded_file is not None:
        raw_smile_df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully.")
    else:
        smile_text = st.text_area(
            "Or paste strike and market price data as CSV",
            value=sample_data,
            height=220,
        )
        raw_smile_df = pd.read_csv(StringIO(smile_text))

    st.markdown("### Market Data Preview")
    st.dataframe(raw_smile_df, use_container_width=True)

    st.markdown("### Smile Settings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        smile_spot = st.number_input(
            "Spot price",
            value=100.0,
            min_value=0.01,
            step=1.0,
            key="real_smile_spot",
        )

    with col2:
        smile_maturity = st.number_input(
            "Time to maturity",
            value=1.0,
            min_value=0.001,
            step=0.01,
            key="real_smile_maturity",
        )

    with col3:
        smile_rate = st.number_input(
            "Risk-free rate",
            value=0.05,
            step=0.001,
            key="real_smile_rate",
        )

    with col4:
        smile_dividend = st.number_input(
            "Dividend yield",
            value=0.0,
            step=0.001,
            key="real_smile_dividend",
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        smile_option_type = st.selectbox(
            "Option type",
            options=["call", "put"],
            key="real_smile_option_type",
        )

    with col6:
        strike_col = st.selectbox(
            "Strike column",
            options=list(raw_smile_df.columns),
            index=list(raw_smile_df.columns).index("strike")
            if "strike" in raw_smile_df.columns
            else 0,
            key="real_smile_strike_col",
        )

    with col7:
        price_col = st.selectbox(
            "Market price column",
            options=list(raw_smile_df.columns),
            index=list(raw_smile_df.columns).index("market_price")
            if "market_price" in raw_smile_df.columns
            else 0,
            key="real_smile_price_col",
        )

    smile_inputs = SmileInputs(
        spot=smile_spot,
        maturity=smile_maturity,
        rate=smile_rate,
        dividend_yield=smile_dividend,
        option_type=smile_option_type,
    )

    try:
        smile_df = build_volatility_smile(
            market_data=raw_smile_df,
            smile_inputs=smile_inputs,
            strike_col=strike_col,
            price_col=price_col,
        )

        valid_df = valid_smile_points(smile_df)

        st.markdown("### Implied Volatility Table")

        display_smile_df = smile_df.copy()

        rounded_cols = [
            "market_price",
            "implied_volatility",
            "implied_volatility_pct",
            "moneyness_K_over_S",
            "log_moneyness",
            "forward",
            "forward_moneyness_K_over_F",
            "lower_bound",
            "upper_bound",
            "pricing_error",
        ]

        for col in rounded_cols:
            if col in display_smile_df.columns:
                display_smile_df[col] = display_smile_df[col].round(6)

        st.dataframe(display_smile_df, use_container_width=True)

        if valid_df.empty:
            st.warning(
                "No valid implied volatility points were found. "
                "Please check the market prices and no-arbitrage bounds."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Valid smile points", f"{len(valid_df)}")
            m2.metric(
                "Average IV",
                f"{valid_df['implied_volatility'].mean():.2%}",
            )
            m3.metric(
                "Minimum IV",
                f"{valid_df['implied_volatility'].min():.2%}",
            )
            m4.metric(
                "Maximum IV",
                f"{valid_df['implied_volatility'].max():.2%}",
            )

            st.markdown("### Volatility Smile Plot")

            x_axis_choice = st.selectbox(
                "X-axis",
                options=[
                    "strike",
                    "moneyness_K_over_S",
                    "forward_moneyness_K_over_F",
                    "log_moneyness",
                ],
                index=0,
                key="smile_x_axis_choice",
            )

            fig_smile = px.line(
                valid_df,
                x=x_axis_choice,
                y="implied_volatility",
                markers=True,
                title="Implied Volatility Smile",
                hover_data=[
                    "strike",
                    "market_price",
                    "implied_volatility_pct",
                    "moneyness_K_over_S",
                    "forward_moneyness_K_over_F",
                ],
            )

            fig_smile.update_layout(
                xaxis_title=x_axis_choice,
                yaxis_title="Implied Volatility",
                hovermode="x unified",
            )

            fig_smile.update_yaxes(tickformat=".1%")

            st.plotly_chart(fig_smile, use_container_width=True)

            st.markdown("### Market Price by Strike")

            fig_price = px.line(
                smile_df,
                x="strike",
                y="market_price",
                markers=True,
                title="Market Option Price by Strike",
            )

            fig_price.update_layout(
                xaxis_title="Strike",
                yaxis_title="Market Option Price",
                hovermode="x unified",
            )

            st.plotly_chart(fig_price, use_container_width=True)

            invalid_df = smile_df[smile_df["status"] != "ok"]

            if not invalid_df.empty:
                st.markdown("### Invalid / Failed Points")
                st.warning(
                    "Some strikes could not be converted into implied volatility. "
                    "This usually happens when prices violate no-arbitrage bounds "
                    "or are inconsistent with the selected option type."
                )
                st.dataframe(
                    invalid_df[["strike", "market_price", "status"]],
                    use_container_width=True,
                )

    except Exception as error:
        st.error(f"Could not build volatility smile: {error}")

with tab_scenario:
    st.subheader("Scenario Analysis")

    st.markdown(
        """
        This section studies how the option price and Greeks change under
        different market scenarios. It is useful for risk analysis, stress
        testing, and understanding option exposure.
        """
    )

    st.markdown("### Base Option Inputs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        scenario_spot = st.number_input(
            "Base spot",
            value=100.0,
            min_value=0.01,
            step=1.0,
            key="scenario_base_spot",
        )
        scenario_strike = st.number_input(
            "Base strike",
            value=100.0,
            min_value=0.01,
            step=1.0,
            key="scenario_base_strike",
        )

    with col2:
        scenario_maturity = st.number_input(
            "Base maturity",
            value=1.0,
            min_value=0.001,
            step=0.01,
            key="scenario_base_maturity",
        )
        scenario_volatility = st.number_input(
            "Base volatility",
            value=0.20,
            min_value=0.001,
            step=0.01,
            key="scenario_base_volatility",
        )

    with col3:
        scenario_rate = st.number_input(
            "Base risk-free rate",
            value=0.05,
            step=0.001,
            key="scenario_base_rate",
        )
        scenario_dividend = st.number_input(
            "Base dividend yield",
            value=0.0,
            step=0.001,
            key="scenario_base_dividend",
        )

    with col4:
        scenario_option_type = st.selectbox(
            "Option type",
            options=["call", "put"],
            key="scenario_base_option_type",
        )

    base_inputs = BlackScholesInputs(
        spot=scenario_spot,
        strike=scenario_strike,
        maturity=scenario_maturity,
        rate=scenario_rate,
        volatility=scenario_volatility,
        dividend_yield=scenario_dividend,
        option_type=scenario_option_type,
    )

    base_result = black_scholes_greeks(base_inputs)

    st.markdown("### Base Case Results")

    metric_cols = st.columns(6)

    metric_cols[0].metric("Price", f"{base_result.price:.4f}")
    metric_cols[1].metric("Delta", f"{base_result.delta:.4f}")
    metric_cols[2].metric("Gamma", f"{base_result.gamma:.4f}")
    metric_cols[3].metric("Vega / 1% vol", f"{base_result.vega:.4f}")
    metric_cols[4].metric("Theta / day", f"{base_result.theta:.4f}")
    metric_cols[5].metric("Rho / 1% rate", f"{base_result.rho:.4f}")

    st.divider()

    tab_sv, tab_greek, tab_time = st.tabs(
        [
            "Spot × Volatility Stress Test",
            "Greek Sensitivity",
            "Time Decay",
        ]
    )

    with tab_sv:
        st.markdown("### Spot × Volatility Stress Test")

        c1, c2, c3 = st.columns(3)

        with c1:
            spot_range = st.slider(
                "Spot shock range (%)",
                min_value=-50,
                max_value=50,
                value=(-20, 20),
                step=5,
                key="scenario_spot_range",
            )

        with c2:
            vol_range = st.slider(
                "Volatility shock range (percentage points)",
                min_value=-50,
                max_value=50,
                value=(-10, 10),
                step=5,
                key="scenario_vol_range",
            )

        with c3:
            shock_step = st.number_input(
                "Shock step",
                value=5,
                min_value=1,
                max_value=25,
                step=1,
                key="scenario_shock_step",
            )

        spot_shocks = make_shock_grid(
            lower_pct=spot_range[0],
            upper_pct=spot_range[1],
            step_pct=shock_step,
        )

        vol_shocks = make_shock_grid(
            lower_pct=vol_range[0],
            upper_pct=vol_range[1],
            step_pct=shock_step,
        )

        scenario_df = build_spot_vol_scenario_table(
            base_inputs=base_inputs,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
        )

        st.markdown("#### Scenario Table")

        display_df = scenario_df.copy()

        round_cols = [
            "spot_shock_pct",
            "vol_shock_pct_points",
            "spot",
            "volatility",
            "volatility_pct",
            "price",
            "price_change",
            "price_change_pct",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        ]

        for col in round_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True)

        heatmap_metric = st.selectbox(
            "Heatmap metric",
            options=[
                "price",
                "price_change",
                "price_change_pct",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
            ],
            key="scenario_heatmap_metric",
        )

        heatmap_df = scenario_df.pivot(
            index="vol_shock_pct_points",
            columns="spot_shock_pct",
            values=heatmap_metric,
        )

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".2f",
            aspect="auto",
            title=f"{heatmap_metric.replace('_', ' ').title()} Heatmap",
        )

        fig_heatmap.update_layout(
            xaxis_title="Spot Shock (%)",
            yaxis_title="Volatility Shock (percentage points)",
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab_greek:
        st.markdown("### Greek Sensitivity by Spot Price")

        spot_min = st.number_input(
            "Minimum spot for sensitivity",
            value=float(scenario_spot * 0.5),
            min_value=0.01,
            step=1.0,
            key="greek_sensitivity_min_spot",
        )

        spot_max = st.number_input(
            "Maximum spot for sensitivity",
            value=float(scenario_spot * 1.5),
            min_value=0.01,
            step=1.0,
            key="greek_sensitivity_max_spot",
        )

        if spot_min >= spot_max:
            st.error("Minimum spot must be smaller than maximum spot.")
        else:
            spot_grid = np.linspace(spot_min, spot_max, 101)

            greek_df = build_greek_sensitivity_table(
                base_inputs=base_inputs,
                spot_grid=spot_grid,
            )

            selected_metrics = st.multiselect(
                "Select metrics to plot",
                options=["price", "delta", "gamma", "vega", "theta", "rho"],
                default=["price", "delta", "gamma"],
                key="greek_sensitivity_metrics",
            )

            if selected_metrics:
                greek_long = greek_df.melt(
                    id_vars=["spot", "moneyness_S_over_K"],
                    value_vars=selected_metrics,
                    var_name="metric",
                    value_name="value",
                )

                fig_greeks = px.line(
                    greek_long,
                    x="spot",
                    y="value",
                    color="metric",
                    title="Price and Greek Sensitivity by Spot Price",
                )

                fig_greeks.update_layout(
                    xaxis_title="Spot Price",
                    yaxis_title="Value",
                    hovermode="x unified",
                )

                st.plotly_chart(fig_greeks, use_container_width=True)

            st.dataframe(greek_df.round(6), use_container_width=True)

    with tab_time:
        st.markdown("### Time Decay Analysis")

        min_days = st.number_input(
            "Minimum days to maturity",
            value=1,
            min_value=1,
            max_value=3650,
            step=1,
            key="time_decay_min_days",
        )

        max_days = st.number_input(
            "Maximum days to maturity",
            value=int(max(2, scenario_maturity * 365)),
            min_value=2,
            max_value=3650,
            step=1,
            key="time_decay_max_days",
        )

        if min_days >= max_days:
            st.error("Minimum days must be smaller than maximum days.")
        else:
            days_grid = np.linspace(max_days, min_days, 100)
            maturity_grid = days_grid / 365.0

            time_df = build_time_decay_table(
                base_inputs=base_inputs,
                maturity_grid=maturity_grid,
            )

            selected_time_metric = st.selectbox(
                "Time decay metric",
                options=["price", "delta", "gamma", "vega", "theta", "rho"],
                index=0,
                key="time_decay_metric",
            )

            fig_time = px.line(
                time_df,
                x="days_to_maturity",
                y=selected_time_metric,
                title=f"{selected_time_metric.capitalize()} as Time to Maturity Changes",
            )

            fig_time.update_layout(
                xaxis_title="Days to Maturity",
                yaxis_title=selected_time_metric.capitalize(),
                hovermode="x unified",
            )

            fig_time.update_xaxes(autorange="reversed")

            st.plotly_chart(fig_time, use_container_width=True)

            st.dataframe(time_df.round(6), use_container_width=True)




# ============================================================
# Footer note
# ============================================================

st.caption(
    "Educational and research prototype. "
    "The dashboard uses the Black-Scholes framework with continuous dividend yield."
)