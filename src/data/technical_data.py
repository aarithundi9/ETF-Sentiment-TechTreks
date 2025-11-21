"""
Technical data acquisition module.

This module provides functions to:
- Fetch real historical price data from yfinance (for future use)
- Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Abstract the data source so it can be easily swapped

For now, this serves as a skeleton with yfinance integration ready
but commented out for development.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import TICKERS, START_DATE, END_DATE, TECHNICAL_INDICATORS


class TechnicalDataFetcher:
    """
    Abstraction layer for fetching technical/price data.
    
    This class can be configured to use different data sources:
    - Mock data (default for development)
    - yfinance (for real data)
    - CSV files
    - Database
    """
    
    def __init__(self, source: str = "mock"):
        """
        Initialize the data fetcher.
        
        Args:
            source: Data source type ('mock', 'yfinance', 'csv', 'db')
        """
        self.source = source
    
    def fetch_ohlcv(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for specified tickers.
        
        Args:
            tickers: List of ticker symbols (default: from config)
            start_date: Start date string (default: from config)
            end_date: End date string (default: from config)
            
        Returns:
            DataFrame with OHLCV data
        """
        tickers = tickers or TICKERS
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        
        if self.source == "mock":
            return self._fetch_mock_data()
        elif self.source == "yfinance":
            return self._fetch_yfinance_data(tickers, start_date, end_date)
        elif self.source == "csv":
            return self._fetch_csv_data()
        else:
            raise ValueError(f"Unknown data source: {self.source}")
    
    def _fetch_mock_data(self) -> pd.DataFrame:
        """Load mock price data from CSV."""
        from src.data.mock_data_generator import load_mock_data
        prices_df, _ = load_mock_data()
        return prices_df
    
    def _fetch_yfinance_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch real data from yfinance (skeleton for future use).
        
        Uncomment when ready to use real data.
        """
        # import yfinance as yf
        # 
        # all_data = []
        # for ticker in tickers:
        #     df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        #     df = df.reset_index()
        #     df['ticker'] = ticker
        #     df.columns = [col.lower() for col in df.columns]
        #     all_data.append(df)
        # 
        # return pd.concat(all_data, ignore_index=True)
        
        raise NotImplementedError(
            "yfinance integration is ready but commented out. "
            "Uncomment the code in _fetch_yfinance_data() to use real data."
        )
    
    def _fetch_csv_data(self) -> pd.DataFrame:
        """Load data from user-provided CSV files."""
        from src.data.user_data_loader import load_user_price_data
        return load_user_price_data()


def calculate_sma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages.
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods to calculate (e.g., [5, 10, 20])
        
    Returns:
        DataFrame with added SMA columns
    """
    df = df.copy()
    for period in periods:
        df[f"sma_{period}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
    return df


def calculate_ema(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages.
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods to calculate
        
    Returns:
        DataFrame with added EMA columns
    """
    df = df.copy()
    for period in periods:
        df[f"ema_{period}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)
        
    Returns:
        DataFrame with added RSI column
    """
    df = df.copy()
    
    def compute_rsi(close_prices):
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df["rsi"] = df.groupby("ticker")["close"].transform(compute_rsi)
    return df


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with MACD, signal, and histogram columns
    """
    df = df.copy()
    
    def compute_macd(close_prices):
        ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return pd.DataFrame({
            "macd": macd,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        })
    
    macd_data = df.groupby("ticker")["close"].apply(compute_macd).reset_index(level=0, drop=True)
    df = pd.concat([df, macd_data], axis=1)
    
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'close' column
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        DataFrame with Bollinger Band columns
    """
    df = df.copy()
    
    df["bb_middle"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(window=period, min_periods=1).mean()
    )
    df["bb_std"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(window=period, min_periods=1).std()
    )
    df["bb_upper"] = df["bb_middle"] + (std_dev * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (std_dev * df["bb_std"])
    
    # Drop intermediate column
    df = df.drop(columns=["bb_std"])
    
    return df


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical indicators added
    """
    print("Calculating technical indicators...")
    
    # Sort by ticker and date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Calculate all indicators
    df = calculate_sma(df, TECHNICAL_INDICATORS["sma_periods"])
    df = calculate_ema(df, TECHNICAL_INDICATORS["ema_periods"])
    df = calculate_rsi(df, TECHNICAL_INDICATORS["rsi_period"])
    df = calculate_macd(
        df,
        TECHNICAL_INDICATORS["macd_fast"],
        TECHNICAL_INDICATORS["macd_slow"],
        TECHNICAL_INDICATORS["macd_signal"],
    )
    df = calculate_bollinger_bands(
        df,
        TECHNICAL_INDICATORS["bollinger_period"],
        TECHNICAL_INDICATORS["bollinger_std"],
    )
    
    print(f"✓ Added {len([c for c in df.columns if c not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
    
    return df


if __name__ == "__main__":
    # Test the technical data fetcher
    print("=" * 80)
    print("Testing Technical Data Fetcher")
    print("=" * 80)
    
    # Fetch mock data
    fetcher = TechnicalDataFetcher(source="mock")
    df = fetcher.fetch_ohlcv()
    
    print(f"\nFetched {len(df)} rows of OHLCV data")
    print(f"Tickers: {df['ticker'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Add technical indicators
    df_with_indicators = add_all_technical_indicators(df)
    
    print("\n" + "=" * 80)
    print("Sample data with technical indicators:")
    print("=" * 80)
    print(df_with_indicators.head())
    
    print("\n" + "=" * 80)
    print("Column names:")
    print("=" * 80)
    print(df_with_indicators.columns.tolist())
