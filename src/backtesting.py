"""
VaR Backtesting Module
----------------------
Validates VaR models by comparing forecasts to actual losses.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a VaR backtest."""
    var_type: str
    confidence_level: float
    total_observations: int
    expected_breaches: float
    actual_breaches: int
    breach_rate: float
    expected_breach_rate: float
    status: str  # 'good', 'warning', 'bad'
    breach_dates: List[pd.Timestamp]
    breach_returns: List[float]
    
    def to_dict(self) -> Dict:
        return {
            'var_type': self.var_type,
            'confidence_level': self.confidence_level,
            'total_observations': self.total_observations,
            'expected_breaches': self.expected_breaches,
            'actual_breaches': self.actual_breaches,
            'breach_rate': self.breach_rate,
            'status': self.status
        }


class VaRBacktester:
    """
    Backtests VaR models by checking if actual losses exceeded forecasts.
    
    The idea is simple: if we have a 95% VaR, we expect about 5% of days
    to have losses exceeding the VaR. If we see way more or fewer breaches,
    something might be wrong with the model.
    """
    
    def __init__(
        self, 
        returns: pd.Series,
        var_forecasts: pd.DataFrame,
        confidence_levels: List[float] = [0.95, 0.99]
    ):
        """
        Initialize backtester.
        
        Args:
            returns: Actual portfolio returns.
            var_forecasts: DataFrame with VaR forecasts (columns for each VaR type).
            confidence_levels: List of confidence levels to test.
        """
        # VaR at time t predicts risk for time t+1, so shift by 1
        self.returns = returns.dropna()
        self.var_forecasts = var_forecasts.shift(1).dropna()
        
        # Align to common dates
        common_idx = self.returns.index.intersection(self.var_forecasts.index)
        self.returns = self.returns.loc[common_idx]
        self.var_forecasts = self.var_forecasts.loc[common_idx]
        
        self.confidence_levels = confidence_levels
        self.results: Dict[str, BacktestResult] = {}
        
    def identify_breaches(self, var_column: str) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Identify days where actual loss exceeded VaR forecast.
        
        A breach happens when: actual_return < -VaR_forecast
        """
        var_values = self.var_forecasts[var_column]
        breaches = self.returns < -var_values
        
        breach_details = pd.DataFrame({
            'date': self.returns.index,
            'return': self.returns.values,
            'var_forecast': var_values.values,
            'is_breach': breaches.values,
            'excess_loss': np.where(breaches, -self.returns.values - var_values.values, 0)
        }).set_index('date')
        
        return breaches, breach_details
    
    def get_status(self, actual_breaches: int, expected_breaches: float, confidence: float) -> str:
        """
        Determine if the VaR model is performing acceptably.
        
        Simple logic:
        - If breaches are within 50% of expected: good
        - If breaches are 50-100% off: warning
        - If breaches are more than 100% off: bad
        """
        if expected_breaches == 0:
            return 'good' if actual_breaches == 0 else 'bad'
        
        ratio = actual_breaches / expected_breaches
        
        if 0.5 <= ratio <= 1.5:
            return 'good'
        elif 0.25 <= ratio <= 2.0:
            return 'warning'
        else:
            return 'bad'
    
    def backtest_var(self, var_column: str, confidence: float) -> BacktestResult:
        """
        Run backtest for a specific VaR column.
        """
        breaches, breach_details = self.identify_breaches(var_column)
        
        n_obs = len(breaches)
        n_breaches = breaches.sum()
        expected_breach_rate = 1 - confidence
        expected_breaches = n_obs * expected_breach_rate
        actual_breach_rate = n_breaches / n_obs if n_obs > 0 else 0
        
        status = self.get_status(n_breaches, expected_breaches, confidence)
        
        breach_df = breach_details[breach_details['is_breach']]
        
        result = BacktestResult(
            var_type=var_column,
            confidence_level=confidence,
            total_observations=n_obs,
            expected_breaches=expected_breaches,
            actual_breaches=n_breaches,
            breach_rate=actual_breach_rate,
            expected_breach_rate=expected_breach_rate,
            status=status,
            breach_dates=list(breach_df.index),
            breach_returns=list(breach_df['return'])
        )
        
        self.results[var_column] = result
        return result
    
    def run_all_backtests(self) -> pd.DataFrame:
        """Run backtests for all VaR columns."""
        results_list = []
        
        for col in self.var_forecasts.columns:
            for conf in self.confidence_levels:
                conf_pct = int(conf * 100)
                if f'_{conf_pct}' in col:
                    result = self.backtest_var(col, conf)
                    results_list.append({
                        'var_type': result.var_type,
                        'confidence': f"{int(result.confidence_level * 100)}%",
                        'days': result.total_observations,
                        'expected': result.expected_breaches,
                        'actual': result.actual_breaches,
                        'breach_rate': f"{result.breach_rate:.1%}",
                        'status': result.status
                    })
                    break
        
        return pd.DataFrame(results_list)
    
    def get_breach_timeline(self) -> pd.DataFrame:
        """Get timeline of all VaR breaches."""
        all_breaches = []
        
        for var_type, result in self.results.items():
            for date, ret in zip(result.breach_dates, result.breach_returns):
                all_breaches.append({
                    'date': date,
                    'var_type': var_type,
                    'return': ret,
                    'confidence': result.confidence_level
                })
        
        if not all_breaches:
            return pd.DataFrame()
        
        return pd.DataFrame(all_breaches).sort_values('date')
    
    def generate_summary_report(self) -> str:
        """Generate a simple backtest summary."""
        if not self.results:
            self.run_all_backtests()
        
        report = []
        report.append("=" * 60)
        report.append("VaR BACKTEST SUMMARY")
        report.append("=" * 60)
        
        for var_type, result in self.results.items():
            status_symbol = {'good': '[OK]', 'warning': '[!]', 'bad': '[X]'}[result.status]
            report.append(f"\n{status_symbol} {var_type}")
            report.append(f"    Confidence: {result.confidence_level:.0%}")
            report.append(f"    Expected breaches: {result.expected_breaches:.1f}")
            report.append(f"    Actual breaches: {result.actual_breaches}")
            report.append(f"    Breach rate: {result.breach_rate:.2%}")
        
        # Count statuses
        statuses = [r.status for r in self.results.values()]
        report.append("\n" + "-" * 60)
        report.append(f"Good: {statuses.count('good')} | Warning: {statuses.count('warning')} | Bad: {statuses.count('bad')}")
        
        return "\n".join(report)


def main():
    """Example usage."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    returns = pd.Series(np.random.normal(0.0003, 0.012, 500), index=dates)
    
    # Add some bad days
    for i in [50, 120, 200, 350]:
        returns.iloc[i] = -0.04
    
    # Simulate VaR forecasts
    var_95 = pd.Series(0.018, index=dates, name='hist_var_95')
    var_99 = pd.Series(0.028, index=dates, name='hist_var_99')
    var_forecasts = pd.DataFrame({'hist_var_95': var_95, 'hist_var_99': var_99})
    
    backtester = VaRBacktester(returns, var_forecasts)
    summary = backtester.run_all_backtests()
    
    print(summary.to_string())
    print("\n")
    print(backtester.generate_summary_report())
    
    return backtester


if __name__ == "__main__":
    main()
