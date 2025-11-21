"""
Mock data generator for ETF Sentiment Analysis project.

This module generates realistic mock data for development and testing:
- Historical OHLCV price data for ETFs
- Mock sentiment scores correlated with price movements
- Mock news counts and social metrics

The mock data is designed to simulate real patterns while being reproducible.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    TICKERS,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
    MOCK_DATA_CONFIG,
)


def generate_price_series(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    volatility: float = 0.02,
    drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock OHLCV data using geometric Brownian motion.
    
    Args:
        ticker: ETF ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        initial_price: Starting price
        volatility: Daily volatility
        drift: Daily drift (trend)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    np.random.seed(seed + hash(ticker) % 1000)
    
    # Generate date range (business days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    num_days = len(dates)
    
    # Generate price using geometric Brownian motion
    returns = np.random.normal(drift, volatility, num_days)
    price_multipliers = np.exp(returns)
    
    # Calculate closing prices
    close_prices = initial_price * np.cumprod(price_multipliers)
    
    # Generate OHLC from close prices
    # Open is close from previous day (with small random gap)
    open_prices = np.zeros(num_days)
    open_prices[0] = initial_price
    open_prices[1:] = close_prices[:-1] * (1 + np.random.normal(0, 0.002, num_days - 1))
    
    # High and Low based on intraday volatility
    intraday_range = np.abs(np.random.normal(0, volatility * 0.5, num_days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range)
    low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range)
    
    # Generate volume (millions of shares)
    base_volume = 50_000_000 if ticker == "SPY" else 30_000_000
    volumes = np.random.lognormal(
        np.log(base_volume), 0.3, num_days
    ).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })
    
    return df


def generate_sentiment_series(
    ticker: str,
    price_df: pd.DataFrame,
    correlation: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock sentiment data correlated with price movements.
    
    The sentiment score is designed to have some predictive power:
    - Positive correlation with next-day returns
    - Random noise to make it realistic
    - News counts vary with market activity
    
    Args:
        ticker: ETF ticker symbol
        price_df: DataFrame with price data
        correlation: Correlation between sentiment and next-day returns
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, ticker, sentiment_score, news_count
    """
    np.random.seed(seed + hash(ticker) % 1000 + 100)
    
    # Calculate next-day returns
    price_df = price_df.copy()
    price_df["next_return"] = price_df["close"].pct_change().shift(-1)
    
    num_days = len(price_df)
    
    # Generate base sentiment correlated with future returns
    # Sentiment ranges from -1 (very negative) to +1 (very positive)
    base_sentiment = price_df["next_return"].fillna(0) * correlation
    noise = np.random.normal(0, 0.15, num_days)
    sentiment_scores = np.clip(base_sentiment + noise, -1, 1)
    
    # Generate news counts (more news on volatile days)
    price_df["abs_return"] = np.abs(price_df["close"].pct_change())
    volatility_factor = price_df["abs_return"] / price_df["abs_return"].mean()
    base_news_count = 10
    news_counts = np.random.poisson(
        base_news_count * (1 + volatility_factor), num_days
    )
    
    # Create sentiment DataFrame
    sentiment_df = pd.DataFrame({
        "date": price_df["date"],
        "ticker": ticker,
        "sentiment_score": sentiment_scores,
        "news_count": news_counts,
    })
    
    return sentiment_df


def generate_all_mock_data(
    save: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate all mock data for the project.
    
    Creates:
    - Mock OHLCV price data for all tickers
    - Mock sentiment data correlated with price movements
    
    Args:
        save: If True, save data to CSV files
        verbose: If True, print progress information
        
    Returns:
        Tuple of (prices_df, sentiment_df)
    """
    if verbose:
        print("=" * 80)
        print("Generating Mock Data for ETF Sentiment Analysis")
        print("=" * 80)
        print(f"Tickers: {', '.join(TICKERS)}")
        print(f"Date Range: {START_DATE} to {END_DATE}")
        print(f"Output Directory: {RAW_DATA_DIR}")
        print("-" * 80)
    
    all_prices = []
    all_sentiment = []
    
    for ticker in TICKERS:
        if verbose:
            print(f"\nGenerating data for {ticker}...")
        
        # Generate price data
        initial_price = MOCK_DATA_CONFIG["initial_price"].get(ticker, 100.0)
        price_df = generate_price_series(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            initial_price=initial_price,
            volatility=MOCK_DATA_CONFIG["volatility"],
            drift=MOCK_DATA_CONFIG["drift"],
        )
        all_prices.append(price_df)
        
        if verbose:
            print(f"  Price data: {len(price_df)} days")
            print(f"  Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
        
        # Generate sentiment data
        sentiment_df = generate_sentiment_series(
            ticker=ticker,
            price_df=price_df,
            correlation=0.3,
        )
        all_sentiment.append(sentiment_df)
        
        if verbose:
            print(f"  Sentiment range: {sentiment_df['sentiment_score'].min():.2f} to {sentiment_df['sentiment_score'].max():.2f}")
            print(f"  Avg news count: {sentiment_df['news_count'].mean():.1f}")
    
    # Combine all tickers
    prices_combined = pd.concat(all_prices, ignore_index=True)
    sentiment_combined = pd.concat(all_sentiment, ignore_index=True)
    
    # Save to CSV
    if save:
        prices_path = RAW_DATA_DIR / "mock_prices.csv"
        sentiment_path = RAW_DATA_DIR / "mock_sentiment.csv"
        
        prices_combined.to_csv(prices_path, index=False)
        sentiment_combined.to_csv(sentiment_path, index=False)
        
        if verbose:
            print("\n" + "=" * 80)
            print("✓ Mock data generation complete!")
            print(f"✓ Prices saved to: {prices_path}")
            print(f"✓ Sentiment saved to: {sentiment_path}")
            print(f"  Total records: {len(prices_combined)} price rows, {len(sentiment_combined)} sentiment rows")
            print("=" * 80)
    
    return prices_combined, sentiment_combined


def load_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously generated mock data.
    
    Returns:
        Tuple of (prices_df, sentiment_df)
    """
    prices_path = RAW_DATA_DIR / "mock_prices.csv"
    sentiment_path = RAW_DATA_DIR / "mock_sentiment.csv"
    
    if not prices_path.exists() or not sentiment_path.exists():
        print("Mock data not found. Generating...")
        return generate_all_mock_data()
    
    prices_df = pd.read_csv(prices_path, parse_dates=["date"])
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    
    return prices_df, sentiment_df


if __name__ == "__main__":
    # Generate mock data when run directly
    prices, sentiment = generate_all_mock_data(save=True, verbose=True)
    
    # Show sample data
    print("\n" + "=" * 80)
    print("Sample Price Data:")
    print("=" * 80)
    print(prices.head(10))
    
    print("\n" + "=" * 80)
    print("Sample Sentiment Data:")
    print("=" * 80)
    print(sentiment.head(10))
