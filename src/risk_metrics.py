"""
Risk Metrics Module
-------------------
Core risk calculations: VaR, CVaR, Volatility, EWMA.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Risk metrics calculator for portfolio returns.
    
    Computes:
    - Rolling volatility (simple and EWMA)
    - Value at Risk (Historical and Parametric)
    - Conditional VaR / Expected Shortfall
    - Drawdowns
    """
    
    def __init__(self, returns: pd.Series, confidence_levels: list = [0.95, 0.99]):
        """
        Initialize with return series.
        
        Args:
            returns: Series of returns, indexed by date.
            confidence_levels: List of confidence levels for VaR (e.g., [0.95, 0.99])
        """
        self.returns = returns.dropna()
        self.confidence_levels = confidence_levels
        self.metrics_history: Optional[pd.DataFrame] = None
        
    def rolling_volatility(self, window: int = 20) -> pd.Series:
        """
        Compute rolling standard deviation of returns.
        
        Args:
            window: Rolling window size in days.
            
        Returns:
            Series of rolling volatility values.
        """
        vol = self.returns.rolling(window=window, min_periods=window).std()
        vol.name = f'vol_{window}d'
        return vol
    
    def ewma_volatility(self, span: int = 20, min_periods: int = 20) -> pd.Series:
        """
        Compute Exponentially Weighted Moving Average volatility.
        
        EWMA gives more weight to recent observations, so it reacts 
        faster to volatility changes than simple rolling.
        
        Args:
            span: Span for EWMA decay
            min_periods: Minimum observations before computing.
            
        Returns:
            Series of EWMA volatility values.
        """
        ewma_var = self.returns.pow(2).ewm(span=span, min_periods=min_periods).mean()
        ewma_vol = np.sqrt(ewma_var)
        ewma_vol.name = f'ewma_vol_{span}d'
        return ewma_vol
    
    def historical_var(self, confidence: float = 0.95, window: int = 252) -> pd.Series:
        """
        Historical Simulation VaR.
        
        Uses the empirical quantile of past returns. No assumptions about
        the distribution - just looks at what actually happened.
        
        Args:
            confidence: Confidence level (0.95 = 95% VaR)
            window: Lookback window for historical data.
            
        Returns:
            Series of VaR values (positive number = potential loss)
        """
        alpha = 1 - confidence
        var = self.returns.rolling(window=window, min_periods=window).quantile(alpha)
        var = -var  # Return as positive loss
        var.name = f'hist_var_{int(confidence*100)}'
        return var
    
    def parametric_var(self, confidence: float = 0.95, window: int = 252) -> pd.Series:
        """
        Parametric (Gaussian) VaR.
        
        Assumes returns follow a normal distribution. Uses the mean and
        standard deviation to estimate VaR.
        
        VaR = -mean + std * z_score
        
        Args:
            confidence: Confidence level (0.95 = 95% VaR)
            window: Rolling window for parameter estimation.
            
        Returns:
            Series of VaR values (positive number = potential loss)
        """
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)
        
        rolling_mean = self.returns.rolling(window=window, min_periods=window).mean()
        rolling_std = self.returns.rolling(window=window, min_periods=window).std()
        
        var = -(rolling_mean + z_score * rolling_std)
        var.name = f'param_var_{int(confidence*100)}'
        return var
    
    def ewma_var(self, confidence: float = 0.95, span: int = 94, min_periods: int = 20) -> pd.Series:
        """
        EWMA-based VaR.
        
        Uses EWMA volatility instead of simple rolling volatility.
        Reacts faster to recent volatility changes.
        
        Args:
            confidence: Confidence level.
            span: EWMA span parameter.
            min_periods: Minimum periods before computing.
            
        Returns:
            Series of VaR values.
        """
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)
        
        ewma_vol = self.ewma_volatility(span=span, min_periods=min_periods)
        var = -z_score * ewma_vol
        var.name = f'ewma_var_{int(confidence*100)}'
        return var
    
    def historical_cvar(self, confidence: float = 0.95, window: int = 252) -> pd.Series:
        """
        Historical Conditional VaR (Expected Shortfall).
        
        Average loss on days when loss exceeds VaR.
        Answers: "When things go bad, how bad do they get on average?"
        
        Args:
            confidence: Confidence level.
            window: Lookback window.
            
        Returns:
            Series of CVaR values (positive = potential loss)
        """
        alpha = 1 - confidence
        
        def cvar_calc(returns_window):
            if len(returns_window) < window:
                return np.nan
            var_threshold = np.percentile(returns_window, alpha * 100)
            tail_losses = returns_window[returns_window <= var_threshold]
            if len(tail_losses) == 0:
                return np.nan
            return -tail_losses.mean()
        
        cvar = self.returns.rolling(window=window).apply(cvar_calc, raw=False)
        cvar.name = f'hist_cvar_{int(confidence*100)}'
        return cvar
    
    def parametric_cvar(self, confidence: float = 0.95, window: int = 252) -> pd.Series:
        """
        Parametric (Gaussian) CVaR.
        
        Uses the normal distribution formula for expected shortfall.
        
        Args:
            confidence: Confidence level.
            window: Rolling window.
            
        Returns:
            Series of CVaR values.
        """
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(alpha)
        pdf_z = stats.norm.pdf(z_alpha)
        
        rolling_mean = self.returns.rolling(window=window, min_periods=window).mean()
        rolling_std = self.returns.rolling(window=window, min_periods=window).std()
        
        cvar = -(rolling_mean - rolling_std * pdf_z / alpha)
        cvar.name = f'param_cvar_{int(confidence*100)}'
        return cvar
    
    def drawdowns(self) -> pd.DataFrame:
        """
        Compute drawdown metrics.
        
        Drawdown = how far we've fallen from the peak.
        
        Returns:
            DataFrame with cumulative return, running max, drawdown.
        """
        cumulative = self.returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        
        return pd.DataFrame({
            'cumulative_return': cumulative,
            'running_max': running_max,
            'drawdown': drawdown
        })
    
    def max_drawdown(self) -> float:
        """Return the maximum drawdown (most negative value)."""
        dd = self.drawdowns()
        return dd['drawdown'].min()
    
    def compute_all_metrics(
        self,
        vol_windows: list = [20, 60],
        var_window: int = 252,
        ewma_span: int = 94
    ) -> pd.DataFrame:
        """
        Compute all risk metrics for each day.
        
        Args:
            vol_windows: List of windows for rolling volatility.
            var_window: Window for VaR/CVaR calculation.
            ewma_span: Span for EWMA calculations.
            
        Returns:
            DataFrame with all metrics indexed by date.
        """
        metrics = pd.DataFrame(index=self.returns.index)
        metrics['return'] = self.returns
        
        # Volatility measures
        for window in vol_windows:
            metrics[f'vol_{window}d'] = self.rolling_volatility(window)
            metrics[f'vol_{window}d_annual'] = metrics[f'vol_{window}d'] * np.sqrt(252)
        
        # EWMA volatility
        metrics[f'ewma_vol_{ewma_span}d'] = self.ewma_volatility(span=ewma_span)
        metrics[f'ewma_vol_{ewma_span}d_annual'] = metrics[f'ewma_vol_{ewma_span}d'] * np.sqrt(252)
        
        # VaR and CVaR for each confidence level
        for conf in self.confidence_levels:
            conf_pct = int(conf * 100)
            
            # Historical VaR/CVaR
            metrics[f'hist_var_{conf_pct}'] = self.historical_var(conf, var_window)
            metrics[f'hist_cvar_{conf_pct}'] = self.historical_cvar(conf, var_window)
            
            # Parametric VaR/CVaR
            metrics[f'param_var_{conf_pct}'] = self.parametric_var(conf, var_window)
            metrics[f'param_cvar_{conf_pct}'] = self.parametric_cvar(conf, var_window)
            
            # EWMA VaR
            metrics[f'ewma_var_{conf_pct}'] = self.ewma_var(conf, span=ewma_span)
        
        # Drawdowns
        dd = self.drawdowns()
        metrics['drawdown'] = dd['drawdown']
        
        self.metrics_history = metrics
        logger.info(f"Computed {len(metrics.columns)} risk metrics for {len(metrics)} days")
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent risk metrics."""
        if self.metrics_history is None:
            self.compute_all_metrics()
        
        latest = self.metrics_history.iloc[-1].to_dict()
        return {k: v for k, v in latest.items() if not pd.isna(v)}


class AssetRiskMetrics:
    """Computes risk metrics for individual assets in a portfolio."""
    
    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns.dropna()
        self.weights = weights.reindex(returns.columns).fillna(0)
        
    def correlation_matrix(self, window: int = 60) -> pd.DataFrame:
        """Compute correlation matrix using recent data."""
        return self.returns.tail(window).corr()
    
    def covariance_matrix(self, window: int = 60) -> pd.DataFrame:
        """Compute covariance matrix using recent data."""
        return self.returns.tail(window).cov()
    
    def portfolio_volatility(self, window: int = 60) -> float:
        """Compute portfolio volatility using covariance matrix."""
        cov = self.covariance_matrix(window)
        w = self.weights.values
        variance = w @ cov.values @ w
        return np.sqrt(variance)


def main():
    """Example usage of risk metrics."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='B')
    returns = pd.Series(np.random.normal(0.0003, 0.015, 1000), index=dates, name='portfolio')
    
    rm = RiskMetrics(returns)
    metrics = rm.compute_all_metrics()
    
    print(f"\nComputed metrics shape: {metrics.shape}")
    print(f"Max Drawdown: {rm.max_drawdown():.4%}")
    
    return rm


if __name__ == "__main__":
    main()
