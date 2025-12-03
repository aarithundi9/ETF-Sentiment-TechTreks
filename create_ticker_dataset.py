"""
Create combined dataset for any ETF ticker.

This script loads price and sentiment data from the raw directory,
processes them through the feature engineering pipeline, and saves the
combined dataset similar to QQQtotal.csv.

Usage:
    python create_ticker_dataset.py XLY
    python create_ticker_dataset.py QQQ
    python create_ticker_dataset.py XLI
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.features.build_features import (
    merge_price_and_sentiment,
    create_lagged_features,
    create_target_variable,
)
from src.data.technical_data import add_all_technical_indicators
from src.data.sentiment_data import calculate_sentiment_features
from src.config.settings import FEATURE_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR


def create_ticker_dataset(ticker: str):
    """Create the complete dataset with features for any ticker.
    
    Args:
        ticker: Ticker symbol (e.g., 'XLY', 'QQQ', 'XLI', 'XLF')
    """
    print("=" * 80)
    print(f"Creating {ticker} Total Dataset")
    print("=" * 80)
    
    # Step 1: Load price data
    print(f"\n[1/7] Loading {ticker} price data...")
    price_path = RAW_DATA_DIR / f"yfinance_prices_{ticker}.csv"
    price_df = pd.read_csv(price_path)
    price_df['date'] = pd.to_datetime(price_df['date'])
    print(f"✓ Loaded {len(price_df)} price records from {price_path}")
    
    # Step 2: Load sentiment data
    print(f"\n[2/7] Loading {ticker} sentiment data...")
    sentiment_path = RAW_DATA_DIR / f"gdelt_sentiment_{ticker}.csv"
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    print(f"✓ Loaded {len(sentiment_df)} sentiment records from {sentiment_path}")
    
    # Step 3: Add technical indicators
    print("\n[3/7] Calculating technical indicators...")
    price_df = add_all_technical_indicators(price_df)
    print(f"✓ Added technical indicators")
    
    # Step 4: Add sentiment features
    print("\n[4/7] Calculating sentiment features...")
    sentiment_df = calculate_sentiment_features(sentiment_df)
    print(f"✓ Added sentiment features")
    
    # Step 5: Merge datasets
    print("\n[5/7] Merging price and sentiment data...")
    merged_df = merge_price_and_sentiment(price_df, sentiment_df)
    print(f"✓ Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    # Step 6: Create lagged features
    print("\n[6/7] Creating lagged features...")
    lag_features = ["close", "volume", "sentiment_score", "rsi"]
    merged_df = create_lagged_features(
        merged_df,
        feature_columns=lag_features,
        lags=FEATURE_CONFIG["lookback_periods"],
    )
    print(f"✓ Created lagged features")
    
    # Step 7: Create target variable
    print("\n[7/7] Creating target variable...")
    merged_df = create_target_variable(
        merged_df,
        forward_period=FEATURE_CONFIG["forward_period"],
        target_type="binary",
    )
    
    # Remove rows with missing target (at the end of the series)
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna(subset=["target"])
    print(f"✓ Created target variable (dropped {initial_rows - len(merged_df)} rows with missing target)")
    
    # Save final dataset
    output_path = PROCESSED_DATA_DIR / f"{ticker}total.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    print(f"  Shape: {merged_df.shape}")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    print("\n" + "=" * 80)
    print(f"✓ {ticker} dataset creation complete!")
    print("=" * 80)
    
    # Show sample
    print("\nFirst 5 rows:")
    print(merged_df.head())
    
    print("\nColumn names:")
    print(merged_df.columns.tolist())
    
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create combined dataset for any ETF ticker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_ticker_dataset.py XLY
  python create_ticker_dataset.py QQQ
  python create_ticker_dataset.py XLI
  python create_ticker_dataset.py XLF
        """
    )
    parser.add_argument(
        'ticker',
        type=str,
        help='Ticker symbol (e.g., XLY, QQQ, XLI, XLF)'
    )
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    df = create_ticker_dataset(ticker)
