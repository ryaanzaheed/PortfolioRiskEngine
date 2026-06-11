"""
Portfolio Risk Dashboard
------------------------
Streamlit dashboard for visualizing portfolio risk metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.risk_engine import PortfolioRiskEngine

# Page config
st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid #2d2d44;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    h1, h2, h3 {
        color: #e8e8e8 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #00d4aa !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0b0 !important;
    }
    
    .stMetric {
        background: rgba(30, 30, 50, 0.6);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 16px;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#00d4aa',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'light': '#e8e8e8',
    'grid': '#2d2d44'
}


def get_plotly_layout(title: str = "", height: int = 400) -> dict:
    """Standard Plotly layout."""
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(20,20,40,0.5)',
        'title': {'text': title, 'font': {'size': 16, 'color': '#ffffff'}},
        'font': {'color': '#ffffff'},
        'height': height,
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
        'xaxis': {'gridcolor': COLORS['grid'], 'tickfont': {'color': '#ffffff'}},
        'yaxis': {'gridcolor': COLORS['grid'], 'tickfont': {'color': '#ffffff'}},
        'legend': {'bgcolor': 'rgba(20,20,40,0.95)', 'font': {'color': '#ffffff'}}
    }


@st.cache_resource
def load_risk_engine():
    """Load and cache the risk engine."""
    engine = PortfolioRiskEngine(
        portfolio_path="data/portfolio.csv",
        portfolio_value=1_000_000,
        confidence_levels=[0.95, 0.99]
    )
    engine.initialize(years=5, use_cache=True)
    engine.run_full_analysis()
    return engine


def render_sidebar():
    """Render sidebar."""
    st.sidebar.markdown("## Risk Engine")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Portfolio")
    engine = load_risk_engine()
    
    for _, row in engine.data.portfolio.iterrows():
        col1, col2 = st.sidebar.columns([2, 1])
        col1.markdown(f"**{row['ticker']}**")
        col2.markdown(f"{row['weight']:.0%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Info")
    st.sidebar.markdown(f"**Start:** {engine.data.prices.index[0].strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**End:** {engine.data.prices.index[-1].strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Days:** {len(engine.data.prices):,}")


def render_overview_tab(engine):
    """Render overview tab."""
    st.markdown("## Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current = engine.risk_metrics.get_current_metrics()
    portfolio_value = engine.get_portfolio_value_series()
    
    with col1:
        pnl = portfolio_value.iloc[-1] - engine.portfolio_value
        pnl_pct = pnl / engine.portfolio_value
        st.metric("Portfolio Value", f"${portfolio_value.iloc[-1]:,.0f}", f"{pnl_pct:+.2%}")
    
    with col2:
        vol = current.get('vol_20d', 0)
        st.metric("20D Volatility", f"{vol:.2%}", f"Ann: {vol * np.sqrt(252):.1%}")
    
    with col3:
        var95 = current.get('hist_var_95', 0)
        st.metric("VaR (95%)", f"{var95:.2%}", f"${var95 * engine.portfolio_value:,.0f}")
    
    with col4:
        var99 = current.get('hist_var_99', 0)
        st.metric("VaR (99%)", f"{var99:.2%}", f"${var99 * engine.portfolio_value:,.0f}")
    
    with col5:
        dd = current.get('drawdown', 0)
        st.metric("Drawdown", f"{dd:.2%}", f"Max: {engine.risk_metrics.max_drawdown():.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line={'color': COLORS['primary'], 'width': 2},
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(**get_plotly_layout("Portfolio Value Over Time", 350))
        fig.update_layout(yaxis_tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        metrics = engine.metrics_df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['vol_20d'] * np.sqrt(252), name='20D Vol', line={'color': COLORS['success']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['vol_60d'] * np.sqrt(252), name='60D Vol', line={'color': COLORS['warning']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['ewma_vol_94d'] * np.sqrt(252), name='EWMA Vol', line={'color': COLORS['info']}))
        fig.update_layout(**get_plotly_layout("Rolling Volatility (Annualized)", 350))
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dd_data = engine.risk_metrics.drawdowns()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd_data.index, y=dd_data['drawdown'],
            mode='lines', name='Drawdown',
            line={'color': COLORS['danger']},
            fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        fig.update_layout(**get_plotly_layout("Drawdown", 300))
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        portfolio = engine.data.portfolio
        colors = ['#667eea', '#00d4aa', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c', '#e91e63', '#00bcd4', '#ff5722']
        fig = go.Figure(data=[go.Pie(
            labels=portfolio['ticker'],
            values=portfolio['weight'],
            hole=0.4,
            marker={'colors': colors[:len(portfolio)]},
            textinfo='label+percent',
            textfont={'color': 'white'}
        )])
        fig.update_layout(**get_plotly_layout("Portfolio Allocation", 300))
        st.plotly_chart(fig, use_container_width=True)


def render_var_tab(engine):
    """Render VaR/CVaR tab."""
    st.markdown("## Value at Risk Analysis")
    
    metrics = engine.metrics_df
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_var_95'], name='Historical VaR 95%', line={'color': COLORS['primary']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['param_var_95'], name='Parametric VaR 95%', line={'color': COLORS['success']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['ewma_var_95'], name='EWMA VaR 95%', line={'color': COLORS['warning']}))
        fig.update_layout(**get_plotly_layout("VaR Comparison (95%)", 350))
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_var_99'], name='Historical VaR 99%', line={'color': COLORS['primary']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['param_var_99'], name='Parametric VaR 99%', line={'color': COLORS['success']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['ewma_var_99'], name='EWMA VaR 99%', line={'color': COLORS['warning']}))
        fig.update_layout(**get_plotly_layout("VaR Comparison (99%)", 350))
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### VaR vs CVaR (Expected Shortfall)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_cvar_95'], name='CVaR 95%', line={'color': COLORS['danger']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_var_95'], name='VaR 95%', line={'color': COLORS['primary'], 'dash': 'dash'}))
        fig.update_layout(**get_plotly_layout("VaR vs CVaR (95%)", 350))
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_cvar_99'], name='CVaR 99%', line={'color': COLORS['danger']}))
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics['hist_var_99'], name='VaR 99%', line={'color': COLORS['primary'], 'dash': 'dash'}))
        fig.update_layout(**get_plotly_layout("VaR vs CVaR (99%)", 350))
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Current VaR Summary")
    current = engine.risk_metrics.get_current_metrics()
    
    var_data = {
        'Metric': ['Historical VaR', 'Parametric VaR', 'EWMA VaR', 'Historical CVaR', 'Parametric CVaR'],
        '95% (%)': [f"{current.get('hist_var_95', 0):.2%}", f"{current.get('param_var_95', 0):.2%}", f"{current.get('ewma_var_95', 0):.2%}", f"{current.get('hist_cvar_95', 0):.2%}", f"{current.get('param_cvar_95', 0):.2%}"],
        '95% ($)': [f"${current.get('hist_var_95', 0) * engine.portfolio_value:,.0f}", f"${current.get('param_var_95', 0) * engine.portfolio_value:,.0f}", f"${current.get('ewma_var_95', 0) * engine.portfolio_value:,.0f}", f"${current.get('hist_cvar_95', 0) * engine.portfolio_value:,.0f}", f"${current.get('param_cvar_95', 0) * engine.portfolio_value:,.0f}"],
        '99% (%)': [f"{current.get('hist_var_99', 0):.2%}", f"{current.get('param_var_99', 0):.2%}", f"{current.get('ewma_var_99', 0):.2%}", f"{current.get('hist_cvar_99', 0):.2%}", f"{current.get('param_cvar_99', 0):.2%}"],
        '99% ($)': [f"${current.get('hist_var_99', 0) * engine.portfolio_value:,.0f}", f"${current.get('param_var_99', 0) * engine.portfolio_value:,.0f}", f"${current.get('ewma_var_99', 0) * engine.portfolio_value:,.0f}", f"${current.get('hist_cvar_99', 0) * engine.portfolio_value:,.0f}", f"${current.get('param_cvar_99', 0) * engine.portfolio_value:,.0f}"]
    }
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)


def render_backtest_tab(engine):
    """Render backtesting tab."""
    st.markdown("## VaR Backtesting")
    st.markdown("*Checking if our VaR predictions actually worked*")
    
    results = engine.backtest_results
    
    st.markdown("### Results Summary")
    cols = st.columns(min(len(results), 6))  # Max 6 columns
    for i, (_, row) in enumerate(results.iterrows()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Get status - handle both old and new column names
            status = row.get('status', row.get('traffic_light', 'good'))
            var_name = row.get('var_type', str(row.name))
            
            if status == 'good' or status == 'green':
                st.success(f"{var_name}")
            elif status == 'warning' or status == 'yellow':
                st.warning(f"{var_name}")
            else:
                st.error(f"{var_name}")
            
            expected = row.get('expected', row.get('expected_breaches', 0))
            actual = row.get('actual', row.get('actual_breaches', 0))
            breach_rate = row.get('breach_rate', '0%')
            
            st.markdown(f"**Breaches:** {actual:.0f} / {expected:.1f}")
            st.markdown(f"**Rate:** {breach_rate}")
    
    st.markdown("---")
    st.markdown("### Detailed Results")
    st.dataframe(results, use_container_width=True, hide_index=True)
    
    st.markdown("### Returns vs VaR")
    metrics = engine.metrics_df
    breaches = engine.get_breach_timeline()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics.index, y=metrics['return'], mode='lines', name='Daily Return', line={'color': COLORS['light'], 'width': 1}, opacity=0.5))
    fig.add_trace(go.Scatter(x=metrics.index, y=-metrics['hist_var_95'], mode='lines', name='-VaR 95%', line={'color': COLORS['warning'], 'dash': 'dash'}))
    fig.add_trace(go.Scatter(x=metrics.index, y=-metrics['hist_var_99'], mode='lines', name='-VaR 99%', line={'color': COLORS['danger'], 'dash': 'dash'}))
    
    if len(breaches) > 0:
        breach_95 = breaches[breaches['var_type'] == 'hist_var_95']
        breach_99 = breaches[breaches['var_type'] == 'hist_var_99']
        if len(breach_95) > 0:
            fig.add_trace(go.Scatter(x=breach_95['date'], y=breach_95['return'], mode='markers', name='95% Breach', marker={'color': COLORS['warning'], 'size': 8, 'symbol': 'x'}))
        if len(breach_99) > 0:
            fig.add_trace(go.Scatter(x=breach_99['date'], y=breach_99['return'], mode='markers', name='99% Breach', marker={'color': COLORS['danger'], 'size': 10, 'symbol': 'x'}))
    
    fig.update_layout(**get_plotly_layout("Returns vs VaR Threshold", 400))
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)


def render_stress_tab(engine):
    """Render stress testing tab."""
    st.markdown("## Stress Testing")
    st.markdown("*What happens in extreme market conditions?*")
    
    col1, col2, col3 = st.columns(3)
    
    all_pnls = [r.portfolio_pnl_pct for r in engine.stress_tester.results]
    
    with col1:
        worst = min(all_pnls)
        st.metric("Worst Scenario", f"{worst:.1%}", f"${worst * engine.portfolio_value:,.0f}")
    
    with col2:
        avg_loss = np.mean([p for p in all_pnls if p < 0])
        st.metric("Avg Scenario Loss", f"{avg_loss:.1%}")
    
    with col3:
        st.metric("Scenarios Tested", len(all_pnls))
    
    st.markdown("---")
    
    scenarios = [r.scenario_name for r in engine.stress_tester.results]
    pnls = [r.portfolio_pnl_pct for r in engine.stress_tester.results]
    colors = [COLORS['danger'] if p < 0 else COLORS['success'] for p in pnls]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=scenarios, x=pnls, orientation='h',
        marker={'color': colors},
        text=[f"{p:.1%}" for p in pnls],
        textposition='outside',
        textfont={'color': '#ffffff'}
    ))
    fig.update_layout(**get_plotly_layout("Stress Scenario Impact", max(400, len(scenarios) * 35)))
    fig.update_layout(xaxis_tickformat='.0%', yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Scenario Details")
    # Fix 'N/A' values in Recovery (days) column for display
    display_results = engine.stress_results.copy()
    if 'Recovery (days)' in display_results.columns:
        display_results['Recovery (days)'] = display_results['Recovery (days)'].replace('N/A', None)
    st.dataframe(display_results, use_container_width=True, hide_index=True)


def render_attribution_tab(engine):
    """Render risk attribution tab."""
    st.markdown("## Risk Attribution")
    st.markdown("*Which assets contribute most to portfolio risk?*")
    
    attribution = engine.attribution_df
    
    col1, col2, col3 = st.columns(3)
    
    concentration = engine.risk_attribution.concentration_risk()
    div_ratio = engine.risk_attribution.diversification_ratio()
    
    with col1:
        st.metric("Diversification Ratio", f"{div_ratio:.2f}", "Higher = better")
    
    with col2:
        st.metric("Top 3 Concentration", f"{concentration['top_3_concentration']:.1%}")
    
    with col3:
        st.metric("Effective Contributors", f"{concentration['effective_contributors']:.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        colors = ['#667eea', '#00d4aa', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c', '#e91e63', '#00bcd4', '#ff5722']
        fig = go.Figure(data=[go.Pie(
            labels=attribution.index,
            values=attribution['pct_contribution'].clip(lower=0),
            hole=0.4,
            marker={'colors': colors[:len(attribution)]},
            textinfo='label+percent',
            textfont={'color': 'white'}
        )])
        fig.update_layout(**get_plotly_layout("Risk Contribution by Asset", 350))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Weight', x=attribution.index, y=attribution['weight'], marker={'color': COLORS['primary']}))
        fig.add_trace(go.Bar(name='Risk Contribution', x=attribution.index, y=attribution['pct_contribution'], marker={'color': COLORS['danger']}))
        fig.update_layout(**get_plotly_layout("Weight vs Risk", 350))
        fig.update_layout(barmode='group', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Attribution")
    display_attr = attribution.copy()
    display_attr['weight'] = display_attr['weight'].apply(lambda x: f"{x:.1%}")
    display_attr['volatility'] = display_attr['volatility'].apply(lambda x: f"{x:.2%}")
    display_attr['pct_contribution'] = display_attr['pct_contribution'].apply(lambda x: f"{x:.1%}")
    display_attr['component_var_$'] = display_attr['component_var_$'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_attr[['weight', 'volatility', 'pct_contribution', 'component_var_$']], use_container_width=True)


def main():
    """Main dashboard function."""
    render_sidebar()
    
    st.markdown("# Portfolio Risk Engine")
    st.markdown("<span style='color: #ffffff;'>*Real-time risk analytics for multi-asset portfolios*</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        engine = load_risk_engine()
    except Exception as e:
        st.error(f"Error loading risk engine: {e}")
        st.info("Make sure to run data ingestion first: `python run.py`")
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "VaR/CVaR", "Backtesting", "Stress Tests", "Risk Attribution"])
    
    with tab1:
        render_overview_tab(engine)
    
    with tab2:
        render_var_tab(engine)
    
    with tab3:
        render_backtest_tab(engine)
    
    with tab4:
        render_stress_tab(engine)
    
    with tab5:
        render_attribution_tab(engine)
    
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    main()
