"""
Portfolio Risk Engine
---------------------
Main module that ties everything together.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import json

from .data_ingestion import DataIngestion
from .risk_metrics import RiskMetrics, AssetRiskMetrics
from .backtesting import VaRBacktester
from .stress_testing import StressTester
from .risk_attribution import RiskAttribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioRiskEngine:
    """
    Main risk engine that runs everything.
    
    This is the central piece that:
    - Loads portfolio data
    - Computes VaR, CVaR, volatility
    - Runs backtests to check if VaR is accurate
    - Runs stress tests for extreme scenarios
    - Breaks down risk by asset
    """
    
    def __init__(
        self,
        portfolio_path: str = "data/portfolio.csv",
        portfolio_value: float = 1_000_000,
        confidence_levels: list = [0.95, 0.99]
    ):
        """
        Initialize the risk engine.
        
        Args:
            portfolio_path: Path to portfolio CSV file.
            portfolio_value: Portfolio notional value.
            confidence_levels: VaR confidence levels.
        """
        self.portfolio_path = portfolio_path
        self.portfolio_value = portfolio_value
        self.confidence_levels = confidence_levels
        
        # Components (initialized later)
        self.data: Optional[DataIngestion] = None
        self.risk_metrics: Optional[RiskMetrics] = None
        self.backtester: Optional[VaRBacktester] = None
        self.stress_tester: Optional[StressTester] = None
        self.risk_attribution: Optional[RiskAttribution] = None
        
        # Results
        self.metrics_df: Optional[pd.DataFrame] = None
        self.backtest_results: Optional[pd.DataFrame] = None
        self.stress_results: Optional[pd.DataFrame] = None
        self.attribution_df: Optional[pd.DataFrame] = None
        
    def initialize(self, years: int = 5, use_cache: bool = True) -> None:
        """
        Initialize with data.
        
        Args:
            years: Years of historical data to fetch.
            use_cache: Whether to use cached data if available.
        """
        logger.info("Initializing Portfolio Risk Engine...")
        
        self.data = DataIngestion(self.portfolio_path)
        
        if use_cache and self.data.load_cached_data():
            logger.info("Using cached data")
        else:
            logger.info(f"Fetching {years} years of price data...")
            self.data.fetch_prices(years=years)
            self.data.compute_returns()
            self.data.save_data()
        
        logger.info("Risk Engine initialized successfully")
    
    def compute_risk_metrics(
        self,
        vol_windows: list = [20, 60],
        var_window: int = 252,
        ewma_span: int = 94
    ) -> pd.DataFrame:
        """Compute all risk metrics."""
        if self.data is None or self.data.portfolio_returns is None:
            raise ValueError("Must initialize engine first")
        
        logger.info("Computing risk metrics...")
        
        self.risk_metrics = RiskMetrics(
            self.data.portfolio_returns,
            confidence_levels=self.confidence_levels
        )
        
        self.metrics_df = self.risk_metrics.compute_all_metrics(
            vol_windows=vol_windows,
            var_window=var_window,
            ewma_span=ewma_span
        )
        
        logger.info(f"Computed {len(self.metrics_df.columns)} metrics for {len(self.metrics_df)} days")
        return self.metrics_df
    
    def run_backtest(self) -> pd.DataFrame:
        """Run VaR backtesting."""
        if self.metrics_df is None:
            self.compute_risk_metrics()
        
        logger.info("Running VaR backtest...")
        
        # Get VaR columns (not CVaR)
        var_columns = [col for col in self.metrics_df.columns 
                       if 'var' in col.lower() and 'cvar' not in col.lower()]
        var_df = self.metrics_df[var_columns]
        
        self.backtester = VaRBacktester(
            self.data.portfolio_returns,
            var_df,
            confidence_levels=self.confidence_levels
        )
        
        self.backtest_results = self.backtester.run_all_backtests()
        
        logger.info(f"Backtest complete: {len(self.backtest_results)} VaR models tested")
        return self.backtest_results
    
    def run_stress_tests(self) -> pd.DataFrame:
        """Run all stress tests."""
        if self.data is None or self.data.simple_returns is None:
            raise ValueError("Must initialize engine first")
        
        logger.info("Running stress tests...")
        
        weights = self.data.portfolio.set_index('ticker')['weight']
        
        # Use simple returns for stress testing (correct for P&L)
        self.stress_tester = StressTester(
            self.data.simple_returns,
            weights,
            portfolio_value=self.portfolio_value
        )
        
        self.stress_results = self.stress_tester.run_all_scenarios()
        
        logger.info(f"Stress tests complete: {len(self.stress_results)} scenarios tested")
        return self.stress_results
    
    def compute_risk_attribution(self, confidence: float = 0.95) -> pd.DataFrame:
        """Compute risk attribution."""
        if self.data is None or self.data.returns is None:
            raise ValueError("Must initialize engine first")
        
        logger.info("Computing risk attribution...")
        
        weights = self.data.portfolio.set_index('ticker')['weight']
        
        self.risk_attribution = RiskAttribution(
            self.data.returns,
            weights,
            portfolio_value=self.portfolio_value,
            confidence=confidence
        )
        
        self.attribution_df = self.risk_attribution.get_full_attribution()
        
        logger.info(f"Attribution complete for {len(self.attribution_df)} assets")
        return self.attribution_df
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete risk analysis."""
        logger.info("Running full risk analysis...")
        
        metrics = self.compute_risk_metrics()
        backtest = self.run_backtest()
        stress = self.run_stress_tests()
        attribution = self.compute_risk_attribution()
        
        current = self.risk_metrics.get_current_metrics()
        
        summary = {
            'as_of_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'portfolio_value': self.portfolio_value,
            'data_start': str(self.data.prices.index[0].date()),
            'data_end': str(self.data.prices.index[-1].date()),
            'trading_days': len(self.data.prices),
            'current_metrics': current,
            'backtest_summary': backtest.to_dict('records'),
            'stress_test_count': len(stress),
            'worst_stress_scenario': self.stress_tester.results[
                np.argmin([r.portfolio_pnl_pct for r in self.stress_tester.results])
            ].scenario_name if self.stress_tester.results else None,
            'top_risk_contributor': attribution.index[0] if len(attribution) > 0 else None,
            'diversification_ratio': self.risk_attribution.diversification_ratio()
        }
        
        logger.info("Full analysis complete")
        return summary
    
    def get_portfolio_value_series(self) -> pd.Series:
        """Get portfolio value over time."""
        if self.data is None:
            raise ValueError("Must initialize engine first")
        return self.data.get_portfolio_value(self.portfolio_value)
    
    def get_breach_timeline(self) -> pd.DataFrame:
        """Get VaR breach timeline."""
        if self.backtester is None:
            self.run_backtest()
        return self.backtester.get_breach_timeline()
    
    def export_to_csv(self, output_dir: str = "output") -> None:
        """Export all results to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.metrics_df is not None:
            self.metrics_df.to_csv(output_path / "risk_metrics.csv")
            logger.info(f"Saved risk metrics to {output_path / 'risk_metrics.csv'}")
        
        if self.backtest_results is not None:
            self.backtest_results.to_csv(output_path / "backtest_results.csv", index=False)
            logger.info(f"Saved backtest results to {output_path / 'backtest_results.csv'}")
        
        if self.stress_results is not None:
            self.stress_results.to_csv(output_path / "stress_test_results.csv", index=False)
            logger.info(f"Saved stress test results to {output_path / 'stress_test_results.csv'}")
        
        if self.attribution_df is not None:
            self.attribution_df.to_csv(output_path / "risk_attribution.csv")
            logger.info(f"Saved risk attribution to {output_path / 'risk_attribution.csv'}")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report."""
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO RISK ENGINE - SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("PORTFOLIO")
        report.append("-" * 40)
        report.append(f"Value: ${self.portfolio_value:,.0f}")
        if self.data:
            report.append(f"Assets: {len(self.data.portfolio)}")
            report.append(f"Data: {self.data.prices.index[0].date()} to {self.data.prices.index[-1].date()}")
            report.append(f"Days: {len(self.data.prices)}")
        
        if self.risk_metrics:
            current = self.risk_metrics.get_current_metrics()
            report.append("")
            report.append("CURRENT RISK METRICS")
            report.append("-" * 40)
            
            for key, value in current.items():
                if pd.isna(value):
                    continue
                if 'vol' in key.lower():
                    report.append(f"  {key}: {value:.4%}")
                elif 'var' in key.lower() or 'cvar' in key.lower():
                    report.append(f"  {key}: {value:.4%} (${value * self.portfolio_value:,.0f})")
        
        if self.backtester:
            report.append("")
            report.append(self.backtester.generate_summary_report())
        
        if self.stress_tester:
            report.append("")
            report.append(self.stress_tester.generate_report())
        
        if self.risk_attribution:
            report.append("")
            report.append(self.risk_attribution.generate_report())
        
        return "\n".join(report)


def main():
    """Run the portfolio risk engine."""
    engine = PortfolioRiskEngine(
        portfolio_path="data/portfolio.csv",
        portfolio_value=1_000_000,
        confidence_levels=[0.95, 0.99]
    )
    
    engine.initialize(years=5, use_cache=True)
    summary = engine.run_full_analysis()
    
    print(json.dumps(summary, indent=2, default=str))
    print("\n" + engine.generate_summary_report())
    
    engine.export_to_csv()
    
    return engine


if __name__ == "__main__":
    main()
