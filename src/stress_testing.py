"""
Stress Testing Module
---------------------
Historical and hypothetical stress tests for portfolio risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Defines a stress test scenario."""
    name: str
    description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    asset_shocks: Optional[Dict[str, float]] = None  # For hypothetical scenarios
    is_historical: bool = True


@dataclass
class StressTestResult:
    """Results from a stress test."""
    scenario_name: str
    scenario_description: str
    portfolio_pnl: float  # Portfolio P&L impact (negative = loss)
    portfolio_pnl_pct: float  # As percentage
    asset_pnls: Dict[str, float]  # P&L by asset
    worst_day_return: float
    recovery_days: Optional[int]  # Days to recover (for historical)
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        return {
            'scenario': self.scenario_name,
            'description': self.scenario_description,
            'portfolio_pnl_pct': f"{self.portfolio_pnl_pct:.2%}",
            'worst_day': f"{self.worst_day_return:.2%}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'recovery_days': self.recovery_days
        }


# Pre-defined historical stress scenarios
HISTORICAL_SCENARIOS = [
    StressScenario(
        name="COVID-19 Crash",
        description="February-March 2020 market crash",
        start_date="2020-02-19",
        end_date="2020-03-23"
    ),
    StressScenario(
        name="COVID Recovery Volatility",
        description="High volatility during early COVID recovery",
        start_date="2020-03-23",
        end_date="2020-04-30"
    ),
    StressScenario(
        name="2022 Rate Shock",
        description="Fed rate hike impact - Q1 2022",
        start_date="2022-01-03",
        end_date="2022-03-14"
    ),
    StressScenario(
        name="2022 Bear Market",
        description="Full year 2022 bear market",
        start_date="2022-01-03",
        end_date="2022-10-12"
    ),
    StressScenario(
        name="SVB Crisis",
        description="March 2023 regional banking crisis",
        start_date="2023-03-08",
        end_date="2023-03-24"
    ),
    StressScenario(
        name="Aug 2024 Volatility",
        description="August 2024 global market selloff",
        start_date="2024-07-31",
        end_date="2024-08-05"
    ),
]

# Pre-defined hypothetical stress scenarios
HYPOTHETICAL_SCENARIOS = [
    StressScenario(
        name="Equity Crash -20%",
        description="Major equity market crash, flight to safety",
        asset_shocks={
            'SPY': -0.20, 'QQQ': -0.25, 'IWM': -0.25, 'EFA': -0.20,
            'TLT': 0.08, 'IEF': 0.04, 'GLD': 0.05, 'XLE': -0.30,
            'VNQ': -0.22, 'LQD': -0.05
        },
        is_historical=False
    ),
    StressScenario(
        name="Rate Shock +100bps",
        description="Sharp interest rate increase",
        asset_shocks={
            'SPY': -0.05, 'QQQ': -0.08, 'IWM': -0.06, 'EFA': -0.05,
            'TLT': -0.15, 'IEF': -0.08, 'GLD': -0.03, 'XLE': 0.02,
            'VNQ': -0.10, 'LQD': -0.08
        },
        is_historical=False
    ),
    StressScenario(
        name="Stagflation",
        description="High inflation + slowing growth",
        asset_shocks={
            'SPY': -0.15, 'QQQ': -0.20, 'IWM': -0.18, 'EFA': -0.15,
            'TLT': -0.10, 'IEF': -0.06, 'GLD': 0.12, 'XLE': 0.15,
            'VNQ': -0.12, 'LQD': -0.08
        },
        is_historical=False
    ),
    StressScenario(
        name="Oil Crisis",
        description="Oil price spike +40%",
        asset_shocks={
            'SPY': -0.08, 'QQQ': -0.06, 'IWM': -0.10, 'EFA': -0.08,
            'TLT': 0.03, 'IEF': 0.02, 'GLD': 0.05, 'XLE': 0.25,
            'VNQ': -0.05, 'LQD': -0.03
        },
        is_historical=False
    ),
    StressScenario(
        name="Tech Bubble Pop",
        description="Technology sector crash",
        asset_shocks={
            'SPY': -0.12, 'QQQ': -0.30, 'IWM': -0.15, 'EFA': -0.10,
            'TLT': 0.05, 'IEF': 0.03, 'GLD': 0.03, 'XLE': -0.05,
            'VNQ': -0.08, 'LQD': -0.02
        },
        is_historical=False
    ),
    StressScenario(
        name="Flight to Quality",
        description="Risk-off environment",
        asset_shocks={
            'SPY': -0.10, 'QQQ': -0.12, 'IWM': -0.15, 'EFA': -0.12,
            'TLT': 0.10, 'IEF': 0.06, 'GLD': 0.08, 'XLE': -0.15,
            'VNQ': -0.10, 'LQD': -0.04
        },
        is_historical=False
    ),
    StressScenario(
        name="Correlation Breakdown",
        description="All assets decline simultaneously",
        asset_shocks={
            'SPY': -0.12, 'QQQ': -0.15, 'IWM': -0.14, 'EFA': -0.12,
            'TLT': -0.05, 'IEF': -0.03, 'GLD': -0.04, 'XLE': -0.10,
            'VNQ': -0.12, 'LQD': -0.06
        },
        is_historical=False
    ),
]


class StressTester:
    """
    Runs historical and hypothetical stress tests on a portfolio.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        portfolio_weights: pd.Series,
        portfolio_value: float = 1_000_000
    ):
        """
        Initialize stress tester.
        
        Args:
            returns: DataFrame of asset returns (columns = tickers).
            portfolio_weights: Series of weights indexed by ticker.
            portfolio_value: Current portfolio value for P&L calculation.
        """
        self.returns = returns.dropna()
        self.weights = portfolio_weights.reindex(returns.columns).fillna(0)
        self.portfolio_value = portfolio_value
        self.results: List[StressTestResult] = []
        
    def run_historical_scenario(
        self, 
        scenario: StressScenario
    ) -> Optional[StressTestResult]:
        """
        Run a historical stress scenario.
        
        Args:
            scenario: Historical stress scenario definition.
            
        Returns:
            StressTestResult or None if dates not in data.
        """
        if not scenario.is_historical:
            raise ValueError("Use run_hypothetical_scenario for hypothetical scenarios")
        
        # Filter to scenario dates
        start = pd.to_datetime(scenario.start_date)
        end = pd.to_datetime(scenario.end_date)
        
        mask = (self.returns.index >= start) & (self.returns.index <= end)
        scenario_returns = self.returns.loc[mask]
        
        if len(scenario_returns) == 0:
            logger.warning(f"No data for scenario {scenario.name} ({start} to {end})")
            return None
        
        # Calculate cumulative returns for each asset
        cumulative_returns = (1 + scenario_returns).prod() - 1
        
        # Portfolio P&L
        asset_pnls = {}
        for ticker in self.returns.columns:
            weight = self.weights.get(ticker, 0)
            asset_return = cumulative_returns.get(ticker, 0)
            asset_pnls[ticker] = weight * asset_return * self.portfolio_value
        
        portfolio_return = (cumulative_returns * self.weights).sum()
        portfolio_pnl = portfolio_return * self.portfolio_value
        
        # Worst day
        portfolio_daily = (scenario_returns * self.weights).sum(axis=1)
        worst_day = portfolio_daily.min()
        
        # Max drawdown during scenario
        cum_portfolio = (1 + portfolio_daily).cumprod()
        running_max = cum_portfolio.cummax()
        drawdown = (cum_portfolio - running_max) / running_max
        max_dd = drawdown.min()
        
        # Recovery days (if applicable)
        recovery_days = None
        if portfolio_return < 0:
            # Check if portfolio recovered after end date
            post_scenario = self.returns.loc[self.returns.index > end]
            if len(post_scenario) > 0:
                post_portfolio = (post_scenario * self.weights).sum(axis=1)
                cum_recovery = (1 + post_portfolio).cumprod()
                # Find first day cumulative return exceeds loss
                recovery_threshold = 1 / (1 + portfolio_return)
                recovered = cum_recovery >= recovery_threshold
                if recovered.any():
                    recovery_days = recovered.idxmax()
                    recovery_days = (recovery_days - end).days
        
        result = StressTestResult(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_return,
            asset_pnls=asset_pnls,
            worst_day_return=worst_day,
            recovery_days=recovery_days,
            max_drawdown=max_dd
        )
        
        self.results.append(result)
        return result
    
    def run_hypothetical_scenario(
        self, 
        scenario: StressScenario
    ) -> StressTestResult:
        """
        Run a hypothetical stress scenario.
        
        Args:
            scenario: Hypothetical scenario with asset shocks.
            
        Returns:
            StressTestResult with P&L impacts.
        """
        if scenario.is_historical:
            raise ValueError("Use run_historical_scenario for historical scenarios")
        
        if scenario.asset_shocks is None:
            raise ValueError("Hypothetical scenario must have asset_shocks defined")
        
        asset_pnls = {}
        portfolio_return = 0.0
        
        for ticker in self.returns.columns:
            weight = self.weights.get(ticker, 0)
            shock = scenario.asset_shocks.get(ticker, 0)
            
            asset_pnl = weight * shock * self.portfolio_value
            asset_pnls[ticker] = asset_pnl
            portfolio_return += weight * shock
        
        portfolio_pnl = portfolio_return * self.portfolio_value
        
        result = StressTestResult(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_return,
            asset_pnls=asset_pnls,
            worst_day_return=portfolio_return,  # Single-day shock
            recovery_days=None,
            max_drawdown=portfolio_return if portfolio_return < 0 else 0
        )
        
        self.results.append(result)
        return result
    
    def run_all_historical(self) -> List[StressTestResult]:
        """Run all pre-defined historical scenarios."""
        results = []
        for scenario in HISTORICAL_SCENARIOS:
            result = self.run_historical_scenario(scenario)
            if result:
                results.append(result)
        return results
    
    def run_all_hypothetical(self) -> List[StressTestResult]:
        """Run all pre-defined hypothetical scenarios."""
        results = []
        for scenario in HYPOTHETICAL_SCENARIOS:
            result = self.run_hypothetical_scenario(scenario)
            results.append(result)
        return results
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all stress scenarios and return summary.
        
        Returns:
            DataFrame with all scenario results.
        """
        self.results = []  # Reset
        
        self.run_all_historical()
        self.run_all_hypothetical()
        
        return self.get_results_summary()
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            data.append({
                'Scenario': r.scenario_name,
                'Description': r.scenario_description,
                'Portfolio P&L ($)': f"${r.portfolio_pnl:,.0f}",
                'Portfolio P&L (%)': f"{r.portfolio_pnl_pct:.2%}",
                'Worst Day': f"{r.worst_day_return:.2%}",
                'Max Drawdown': f"{r.max_drawdown:.2%}",
                'Recovery (days)': r.recovery_days if r.recovery_days is not None else None
            })
        
        return pd.DataFrame(data)
    
    def get_asset_breakdown(self, scenario_name: str) -> pd.DataFrame:
        """Get asset-level P&L breakdown for a scenario."""
        for r in self.results:
            if r.scenario_name == scenario_name:
                data = []
                for ticker, pnl in r.asset_pnls.items():
                    weight = self.weights.get(ticker, 0)
                    data.append({
                        'Asset': ticker,
                        'Weight': f"{weight:.1%}",
                        'P&L ($)': f"${pnl:,.0f}",
                        'P&L (%)': f"{pnl / self.portfolio_value:.2%}"
                    })
                return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def find_worst_historical_window(
        self, 
        window_days: int = 5
    ) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
        """
        Find the worst historical window of a given length.
        
        Args:
            window_days: Window size in trading days.
            
        Returns:
            Tuple of (start_date, end_date, cumulative_return)
        """
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        # Rolling cumulative return
        rolling_cum = portfolio_returns.rolling(window_days).sum()
        
        worst_end_idx = rolling_cum.idxmin()
        worst_end_loc = self.returns.index.get_loc(worst_end_idx)
        worst_start_loc = max(0, worst_end_loc - window_days + 1)
        worst_start = self.returns.index[worst_start_loc]
        
        worst_return = rolling_cum.loc[worst_end_idx]
        
        return worst_start, worst_end_idx, worst_return
    
    def custom_scenario(
        self,
        name: str,
        shocks: Dict[str, float],
        description: str = ""
    ) -> StressTestResult:
        """
        Run a custom hypothetical scenario.
        
        Args:
            name: Scenario name.
            shocks: Dictionary of ticker -> shock (e.g., {'SPY': -0.10}).
            description: Scenario description.
            
        Returns:
            StressTestResult
        """
        scenario = StressScenario(
            name=name,
            description=description,
            asset_shocks=shocks,
            is_historical=False
        )
        return self.run_hypothetical_scenario(scenario)
    
    def generate_report(self) -> str:
        """Generate a formatted stress test report."""
        if not self.results:
            self.run_all_scenarios()
        
        report = []
        report.append("=" * 70)
        report.append("STRESS TEST REPORT")
        report.append("=" * 70)
        report.append(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        report.append(f"Number of Scenarios: {len(self.results)}")
        report.append("")
        
        # Separate historical and hypothetical
        historical = [r for r in self.results if r.recovery_days is not None or 
                      any(s.name == r.scenario_name for s in HISTORICAL_SCENARIOS)]
        hypothetical = [r for r in self.results if r.scenario_name in 
                       [s.name for s in HYPOTHETICAL_SCENARIOS]]
        
        if historical:
            report.append("\n" + "=" * 50)
            report.append("HISTORICAL SCENARIOS")
            report.append("=" * 50)
            for r in historical:
                report.append(f"\n{r.scenario_name}")
                report.append(f"  {r.scenario_description}")
                report.append(f"  Portfolio P&L: ${r.portfolio_pnl:,.0f} ({r.portfolio_pnl_pct:.2%})")
                report.append(f"  Worst Day: {r.worst_day_return:.2%}")
                report.append(f"  Max Drawdown: {r.max_drawdown:.2%}")
                if r.recovery_days:
                    report.append(f"  Recovery: {r.recovery_days} days")
        
        if hypothetical:
            report.append("\n" + "=" * 50)
            report.append("HYPOTHETICAL SCENARIOS")
            report.append("=" * 50)
            for r in hypothetical:
                report.append(f"\n{r.scenario_name}")
                report.append(f"  {r.scenario_description}")
                report.append(f"  Portfolio Impact: ${r.portfolio_pnl:,.0f} ({r.portfolio_pnl_pct:.2%})")
        
        # Summary statistics
        report.append("\n" + "=" * 50)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 50)
        
        all_pnls = [r.portfolio_pnl_pct for r in self.results]
        report.append(f"Worst Scenario: {min(all_pnls):.2%}")
        report.append(f"Average Scenario Loss: {np.mean([p for p in all_pnls if p < 0]):.2%}")
        
        worst_scenario = min(self.results, key=lambda r: r.portfolio_pnl_pct)
        report.append(f"Worst Scenario Name: {worst_scenario.scenario_name}")
        
        return "\n".join(report)


def main():
    """Example usage of stress testing."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=1500, freq='B')
    
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT', 'IEF', 'GLD', 'XLE', 'VNQ', 'LQD']
    
    # Correlated returns
    returns_data = {}
    base = np.random.normal(0.0004, 0.012, len(dates))
    
    for ticker in tickers:
        noise = np.random.normal(0, 0.008, len(dates))
        if ticker in ['TLT', 'IEF', 'LQD']:
            returns_data[ticker] = -0.3 * base + noise + 0.0002
        elif ticker == 'GLD':
            returns_data[ticker] = -0.2 * base + noise + 0.0001
        else:
            returns_data[ticker] = 0.8 * base + noise + 0.0003
    
    returns = pd.DataFrame(returns_data, index=dates)
    
    # Add COVID-like crash
    covid_period = (dates >= '2020-02-20') & (dates <= '2020-03-23')
    for ticker in ['SPY', 'QQQ', 'IWM', 'EFA', 'XLE', 'VNQ']:
        returns.loc[covid_period, ticker] -= 0.025
    for ticker in ['TLT', 'IEF', 'GLD']:
        returns.loc[covid_period, ticker] += 0.01
    
    weights = pd.Series({
        'SPY': 0.25, 'QQQ': 0.15, 'IWM': 0.10, 'EFA': 0.10,
        'TLT': 0.15, 'IEF': 0.05, 'GLD': 0.08, 'XLE': 0.05,
        'VNQ': 0.04, 'LQD': 0.03
    })
    
    # Run stress tests
    tester = StressTester(returns, weights, portfolio_value=1_000_000)
    
    # Run all scenarios
    summary = tester.run_all_scenarios()
    print(summary.to_string())
    print("\n")
    print(tester.generate_report())
    
    return tester


if __name__ == "__main__":
    main()
