"""
Risk Attribution Module
-----------------------
Figures out which assets contribute most to portfolio risk.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAttribution:
    """
    Breaks down portfolio risk to see which assets drive it.
    
    Key concepts:
    - Marginal VaR: How much does VaR change if we add $1 to an asset?
    - Component VaR: How much of the total VaR comes from each asset?
    - The sum of all Component VaRs = Total Portfolio VaR
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float = 1_000_000,
        confidence: float = 0.95
    ):
        """
        Initialize risk attribution.
        
        Args:
            returns: DataFrame of asset returns (columns = tickers).
            weights: Series of weights indexed by ticker.
            portfolio_value: Portfolio notional value.
            confidence: VaR confidence level.
        """
        self.returns = returns.dropna()
        self.weights = weights.reindex(returns.columns).fillna(0)
        self.portfolio_value = portfolio_value
        self.confidence = confidence
        self.alpha = 1 - confidence
        self.z_score = stats.norm.ppf(self.alpha)
        
        # Compute covariance matrix
        self.cov_matrix = self.returns.cov()
        self.corr_matrix = self.returns.corr()
        
        self._compute_portfolio_metrics()
        
    def _compute_portfolio_metrics(self) -> None:
        """Compute portfolio-level risk metrics."""
        w = self.weights.values
        cov = self.cov_matrix.values
        
        # Portfolio variance: w' * Cov * w (matrix multiplication)
        self.portfolio_variance = w @ cov @ w
        self.portfolio_volatility = np.sqrt(self.portfolio_variance)
        
        # Portfolio VaR (parametric)
        self.portfolio_var = -self.z_score * self.portfolio_volatility
        self.portfolio_var_dollar = self.portfolio_var * self.portfolio_value
        
    def marginal_var(self) -> pd.Series:
        """
        Compute Marginal VaR for each asset.
        
        Marginal VaR tells you: if I increase my position in this asset
        by a tiny amount, how much does portfolio VaR change?
        """
        w = self.weights.values
        cov = self.cov_matrix.values
        
        # Marginal contribution to volatility
        marginal_vol = (cov @ w) / self.portfolio_volatility
        
        # Marginal VaR
        marginal = -self.z_score * marginal_vol
        return pd.Series(marginal, index=self.returns.columns, name='marginal_var')
    
    def component_var(self) -> pd.Series:
        """
        Compute Component VaR for each asset.
        
        Component VaR = weight * Marginal VaR
        
        This tells you how much each asset contributes to total VaR.
        The sum of all component VaRs equals the portfolio VaR.
        """
        marginal = self.marginal_var()
        component = self.weights * marginal
        component.name = 'component_var'
        return component
    
    def percent_contribution(self) -> pd.Series:
        """
        Compute percentage contribution to VaR.
        
        Answers: "What % of my total risk comes from each asset?"
        """
        component = self.component_var()
        pct_contrib = component / component.sum()
        pct_contrib.name = 'pct_var_contribution'
        return pct_contrib
    
    def beta_to_portfolio(self) -> pd.Series:
        """
        Compute each asset's beta to the portfolio.
        
        Beta > 1 means the asset amplifies portfolio moves.
        Beta < 1 means the asset dampens portfolio moves.
        """
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        betas = {}
        for ticker in self.returns.columns:
            cov_with_portfolio = self.returns[ticker].cov(portfolio_returns)
            beta = cov_with_portfolio / portfolio_returns.var()
            betas[ticker] = beta
        
        return pd.Series(betas, name='beta_to_portfolio')
    
    def diversification_ratio(self) -> float:
        """
        Compute portfolio diversification ratio.
        
        DR = (Weighted average volatility) / (Portfolio volatility)
        
        DR > 1 means diversification is helping reduce risk.
        The higher, the more diversification benefit.
        """
        individual_vol = self.returns.std()
        weighted_avg_vol = (self.weights * individual_vol).sum()
        
        return weighted_avg_vol / self.portfolio_volatility
    
    def get_full_attribution(self) -> pd.DataFrame:
        """
        Get complete risk attribution for all assets.
        """
        marginal = self.marginal_var()
        component = self.component_var()
        pct_contrib = self.percent_contribution()
        betas = self.beta_to_portfolio()
        individual_vol = self.returns.std()
        
        attribution = pd.DataFrame({
            'weight': self.weights,
            'volatility': individual_vol,
            'marginal_var': marginal,
            'component_var': component,
            'pct_contribution': pct_contrib,
            'beta_to_portfolio': betas
        })
        
        attribution['component_var_$'] = attribution['component_var'] * self.portfolio_value
        
        # Sort by contribution (highest first)
        attribution = attribution.sort_values('pct_contribution', ascending=False)
        
        return attribution
    
    def concentration_risk(self) -> Dict[str, float]:
        """
        Check how concentrated the risk is.
        
        Returns metrics about risk concentration.
        """
        pct_contrib = self.percent_contribution()
        
        # Herfindahl-Hirschman Index (lower = more diversified)
        hhi = (pct_contrib ** 2).sum()
        
        # Top 3 assets' share of risk
        top_3 = pct_contrib.nlargest(3).sum()
        
        # Effective number of risk contributors
        effective_n = 1 / hhi if hhi > 0 else len(pct_contrib)
        
        return {
            'hhi': hhi,
            'top_3_concentration': top_3,
            'effective_contributors': effective_n
        }
    
    def generate_report(self) -> str:
        """Generate a risk attribution report."""
        attribution = self.get_full_attribution()
        concentration = self.concentration_risk()
        div_ratio = self.diversification_ratio()
        
        report = []
        report.append("=" * 60)
        report.append("RISK ATTRIBUTION REPORT")
        report.append("=" * 60)
        report.append(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        report.append(f"VaR Confidence: {self.confidence:.0%}")
        report.append("")
        
        report.append("PORTFOLIO RISK")
        report.append("-" * 40)
        report.append(f"Daily Volatility: {self.portfolio_volatility:.4%}")
        report.append(f"Annual Volatility: {self.portfolio_volatility * np.sqrt(252):.2%}")
        report.append(f"VaR ({self.confidence:.0%}): ${self.portfolio_var_dollar:,.0f}")
        report.append(f"Diversification Ratio: {div_ratio:.2f}")
        report.append("")
        
        report.append("TOP RISK CONTRIBUTORS")
        report.append("-" * 40)
        for ticker, row in attribution.head(5).iterrows():
            report.append(
                f"  {ticker:6s} | Weight: {row['weight']:5.1%} | "
                f"Risk: {row['pct_contribution']:5.1%} | "
                f"${row['component_var_$']:,.0f}"
            )
        
        report.append("")
        report.append(f"Top 3 assets = {concentration['top_3_concentration']:.1%} of total risk")
        
        return "\n".join(report)


def main():
    """Example usage."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=750, freq='B')
    
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
    n = len(dates)
    
    # Create correlated returns
    market = np.random.normal(0.0004, 0.012, n)
    returns_data = {
        'SPY': 0.9 * market + np.random.normal(0, 0.005, n),
        'QQQ': 1.1 * market + np.random.normal(0, 0.008, n),
        'TLT': -0.3 * market + np.random.normal(0, 0.006, n),
        'GLD': -0.1 * market + np.random.normal(0, 0.007, n)
    }
    returns = pd.DataFrame(returns_data, index=dates)
    
    weights = pd.Series({'SPY': 0.40, 'QQQ': 0.30, 'TLT': 0.20, 'GLD': 0.10})
    
    attr = RiskAttribution(returns, weights)
    print(attr.generate_report())
    
    return attr


if __name__ == "__main__":
    main()
