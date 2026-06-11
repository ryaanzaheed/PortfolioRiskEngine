"""
Portfolio Risk Engine - Main Entry Point
-----------------------------------------
Run this script to fetch data, compute risk metrics, and export results.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.risk_engine import PortfolioRiskEngine


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Risk Engine - Comprehensive market risk analysis"
    )
    parser.add_argument(
        "--portfolio", 
        type=str, 
        default="data/portfolio.csv",
        help="Path to portfolio CSV file"
    )
    parser.add_argument(
        "--value", 
        type=float, 
        default=1_000_000,
        help="Portfolio notional value (default: 1,000,000)"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=5,
        help="Years of historical data to fetch (default: 5)"
    )
    parser.add_argument(
        "--refresh", 
        action="store_true",
        help="Force refresh data (ignore cache)"
    )
    parser.add_argument(
        "--export", 
        type=str, 
        default="output",
        help="Output directory for exports"
    )
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Print detailed report to console"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PORTFOLIO RISK ENGINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio: {args.portfolio}")
    print(f"Notional Value: ${args.value:,.0f}")
    print()
    
    # Initialize engine
    engine = PortfolioRiskEngine(
        portfolio_path=args.portfolio,
        portfolio_value=args.value,
        confidence_levels=[0.95, 0.99]
    )
    
    # Initialize with data
    print("Initializing data...")
    engine.initialize(years=args.years, use_cache=not args.refresh)
    
    # Run full analysis
    print("\nRunning risk analysis...")
    summary = engine.run_full_analysis()
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(json.dumps({
        'as_of_date': summary['as_of_date'],
        'data_start': summary['data_start'],
        'data_end': summary['data_end'],
        'trading_days': summary['trading_days'],
        'portfolio_value': f"${summary['portfolio_value']:,.0f}",
        'worst_stress_scenario': summary['worst_stress_scenario'],
        'top_risk_contributor': summary['top_risk_contributor'],
        'diversification_ratio': f"{summary['diversification_ratio']:.2f}"
    }, indent=2))
    
    # Print current metrics
    print("\nCURRENT RISK METRICS")
    print("-" * 50)
    current = summary['current_metrics']
    
    metrics_to_show = [
        ('vol_20d', 'Volatility (20D)'),
        ('hist_var_95', 'Historical VaR 95%'),
        ('hist_var_99', 'Historical VaR 99%'),
        ('hist_cvar_95', 'Historical CVaR 95%'),
        ('hist_cvar_99', 'Historical CVaR 99%'),
        ('drawdown', 'Current Drawdown')
    ]
    
    for key, label in metrics_to_show:
        if key in current:
            value = current[key]
            dollar = value * engine.portfolio_value
            print(f"  {label:25s}: {value:8.4%}  (${dollar:>12,.0f})")
    
    # Export results
    print("\n" + "-" * 50)
    print("Exporting results...")
    
    # Create output directory
    Path(args.export).mkdir(parents=True, exist_ok=True)
    
    # CSV export
    engine.export_to_csv(args.export)
    
    
    # Print full report if requested
    if args.report:
        print("\n")
        print(engine.generate_summary_report())
    
    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nTo view the dashboard, run:")
    print("  streamlit run dashboard.py")
    
    return engine


if __name__ == "__main__":
    main()
