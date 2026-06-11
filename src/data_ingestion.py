"""
Data Ingestion Module
---------------------
Fetches historical price data and computes returns for portfolio assets.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data fetching and return computation for portfolio assets."""
    
    def __init__(self, portfolio_path: str = "data/portfolio.csv"):
        """
        Initialize with portfolio configuration.
        
        Args:
            portfolio_path: Path to CSV with ticker, weight, asset_class, description
        """
        self.portfolio_path = Path(portfolio_path)
        self.portfolio = self._load_portfolio()
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None  # Log returns
        self.simple_returns: Optional[pd.DataFrame] = None  # Simple returns
        self.portfolio_returns: Optional[pd.Series] = None
        self.portfolio_simple_returns: Optional[pd.Series] = None
        
    def _load_portfolio(self) -> pd.DataFrame:
        """Load portfolio weights from CSV."""
        if not self.portfolio_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {self.portfolio_path}")
        
        df = pd.read_csv(self.portfolio_path)
        
        # Validate weights sum to 1
        weight_sum = df['weight'].sum()
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            logger.warning(f"Portfolio weights sum to {weight_sum:.4f}, normalizing to 1.0")
            df['weight'] = df['weight'] / weight_sum
            
        logger.info(f"Loaded portfolio with {len(df)} assets")
        return df
    
    def fetch_prices(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 5
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for all portfolio assets.
        
        Args:
            start_date: Start date (YYYY-MM-DD). If None, uses 'years' param.
            end_date: End date (YYYY-MM-DD). If None, uses today.
            years: Number of years of history if start_date not specified.
            
        Returns:
            DataFrame with adjusted close prices, indexed by date.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=years * 365)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        tickers = self.portfolio['ticker'].tolist()
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Fetch all tickers at once for efficiency
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,  # Use adjusted prices
            progress=False
        )
        
        # Extract Close prices (already adjusted due to auto_adjust=True)
        if len(tickers) == 1:
            self.prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            self.prices = data['Close']
        
        # Drop any rows with all NaN
        self.prices = self.prices.dropna(how='all')
        
        # Forward fill then backward fill for any remaining gaps
        self.prices = self.prices.ffill().bfill()
        
        logger.info(f"Fetched {len(self.prices)} days of price data")
        return self.prices
    
    def compute_returns(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute daily log returns for each asset and the portfolio.
        Also computes simple returns for P&L/stress testing.
        
        Returns:
            Tuple of (asset_returns DataFrame, portfolio_returns Series)
        """
        if self.prices is None:
            raise ValueError("Must fetch prices first using fetch_prices()")
        
        # Compute log returns (more accurate for risk metrics like VaR)
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        
        # Compute simple returns (correct for P&L and stress testing)
        self.simple_returns = self.prices.pct_change().dropna()
        
        # Compute portfolio returns as weighted sum
        weights = self.portfolio.set_index('ticker')['weight']
        
        # Align weights with returns columns
        aligned_weights = weights.reindex(self.returns.columns)
        
        # Portfolio return = sum of (weight * asset_return)
        self.portfolio_returns = (self.returns * aligned_weights).sum(axis=1)
        self.portfolio_returns.name = 'portfolio_return'
        
        # Simple portfolio returns for P&L calculations
        self.portfolio_simple_returns = (self.simple_returns * aligned_weights).sum(axis=1)
        self.portfolio_simple_returns.name = 'portfolio_simple_return'
        
        logger.info(f"Computed returns for {len(self.returns)} trading days")
        return self.returns, self.portfolio_returns
    
    def compute_simple_returns(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute simple (arithmetic) returns for each asset and portfolio.
        Useful for P&L calculations.
        
        Returns:
            Tuple of (asset_returns DataFrame, portfolio_returns Series)
        """
        if self.prices is None:
            raise ValueError("Must fetch prices first using fetch_prices()")
        
        simple_returns = self.prices.pct_change().dropna()
        
        weights = self.portfolio.set_index('ticker')['weight']
        aligned_weights = weights.reindex(simple_returns.columns)
        
        portfolio_simple_returns = (simple_returns * aligned_weights).sum(axis=1)
        portfolio_simple_returns.name = 'portfolio_return'
        
        return simple_returns, portfolio_simple_returns
    
    def get_portfolio_value(self, initial_value: float = 1_000_000) -> pd.Series:
        """
        Compute portfolio value over time given initial investment.
        
        Args:
            initial_value: Starting portfolio value in dollars.
            
        Returns:
            Series of portfolio values indexed by date.
        """
        if self.portfolio_returns is None:
            raise ValueError("Must compute returns first using compute_returns()")
        
        # Cumulative returns using log returns
        cumulative_returns = self.portfolio_returns.cumsum()
        portfolio_value = initial_value * np.exp(cumulative_returns)
        portfolio_value.name = 'portfolio_value'
        
        return portfolio_value
    
    def get_asset_weights_over_time(self, initial_value: float = 1_000_000) -> pd.DataFrame:
        """
        Track how asset weights drift over time due to price movements.
        
        Returns:
            DataFrame with actual weights for each asset over time.
        """
        if self.prices is None:
            raise ValueError("Must fetch prices first using fetch_prices()")
        
        # Normalize prices to initial = 1
        normalized_prices = self.prices / self.prices.iloc[0]
        
        # Initial dollar allocation per asset
        initial_weights = self.portfolio.set_index('ticker')['weight']
        initial_allocation = initial_value * initial_weights.reindex(self.prices.columns)
        
        # Value of each position over time
        position_values = normalized_prices * initial_allocation
        
        # Total portfolio value
        total_value = position_values.sum(axis=1)
        
        # Actual weights = position value / total value
        actual_weights = position_values.div(total_value, axis=0)
        
        return actual_weights
    
    def save_data(self, output_dir: str = "data") -> None:
        """Save fetched data to CSV files.
        
        Note: simple_returns are NOT saved - they're computed on-the-fly from prices.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.prices is not None:
            self.prices.to_csv(output_path / "prices.csv")
            logger.info(f"Saved prices to {output_path / 'prices.csv'}")
        
        if self.returns is not None:
            self.returns.to_csv(output_path / "returns.csv")
            logger.info(f"Saved returns to {output_path / 'returns.csv'}")
        
        if self.portfolio_returns is not None:
            self.portfolio_returns.to_frame().to_csv(output_path / "portfolio_returns.csv")
            logger.info(f"Saved portfolio returns to {output_path / 'portfolio_returns.csv'}")
    
    def load_cached_data(self, data_dir: str = "data") -> bool:
        """
        Load previously saved price/return data if available.
        Simple returns are computed on-the-fly from prices (not cached).
        
        Returns:
            True if data was loaded, False otherwise.
        """
        data_path = Path(data_dir)
        
        prices_path = data_path / "prices.csv"
        returns_path = data_path / "returns.csv"
        portfolio_returns_path = data_path / "portfolio_returns.csv"
        
        if all(p.exists() for p in [prices_path, returns_path, portfolio_returns_path]):
            self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            self.portfolio_returns = pd.read_csv(
                portfolio_returns_path, index_col=0, parse_dates=True
            ).squeeze()
            
            # Compute simple returns from prices (not cached - derived data)
            self.simple_returns = self.prices.pct_change().dropna()
                
            # Compute simple portfolio returns
            weights = self.portfolio.set_index('ticker')['weight']
            aligned_weights = weights.reindex(self.simple_returns.columns)
            self.portfolio_simple_returns = (self.simple_returns * aligned_weights).sum(axis=1)
            self.portfolio_simple_returns.name = 'portfolio_simple_return'
            
            logger.info("Loaded cached data successfully")
            return True
        
        return False


def main():
    """Example usage of the DataIngestion class."""
    # Initialize with default portfolio
    ingestion = DataIngestion()
    
    # Fetch 5 years of data
    prices = ingestion.fetch_prices(years=5)
    print(f"\nPrice data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Compute returns
    returns, portfolio_returns = ingestion.compute_returns()
    print(f"\nReturns data shape: {returns.shape}")
    print(f"\nPortfolio return statistics:")
    print(f"  Mean daily return: {portfolio_returns.mean():.4%}")
    print(f"  Daily volatility: {portfolio_returns.std():.4%}")
    print(f"  Annualized return: {portfolio_returns.mean() * 252:.2%}")
    print(f"  Annualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    
    # Save data
    ingestion.save_data()
    
    return ingestion


if __name__ == "__main__":
    main()

