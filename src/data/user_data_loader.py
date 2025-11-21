"""
User data loader module.

This module provides functions to load CSV files that users may upload:
- Historical price/OHLCV data
- Sentiment data from external sources
- Any custom features

The loader validates data format and ensures compatibility with the pipeline.
"""

import pandas as pd
from typing import Optional, List
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import RAW_DATA_DIR


def load_user_price_data(
    file_pattern: str = "user_prices*.csv",
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load user-provided price/OHLCV data from CSV files.
    
    Expected format:
    - date: Date column (will be parsed)
    - ticker: Ticker symbol
    - open, high, low, close, volume: OHLCV data
    
    Args:
        file_pattern: Glob pattern to match CSV files
        required_columns: List of required column names
        
    Returns:
        DataFrame with loaded price data
    """
    required_columns = required_columns or [
        "date", "ticker", "open", "high", "low", "close", "volume"
    ]
    
    # Find all matching files
    files = list(RAW_DATA_DIR.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(
            f"No user price data found matching '{file_pattern}' in {RAW_DATA_DIR}"
        )
    
    print(f"Loading user price data from {len(files)} file(s)...")
    
    all_data = []
    for file in files:
        print(f"  Loading {file.name}...")
        df = pd.read_csv(file)
        
        # Validate required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"File {file.name} is missing required columns: {missing_cols}"
            )
        
        # Parse date column
        df["date"] = pd.to_datetime(df["date"])
        
        all_data.append(df)
    
    # Combine all files
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and remove duplicates
    combined_df = combined_df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    ).reset_index(drop=True)
    
    print(f"✓ Loaded {len(combined_df)} rows of user price data")
    
    return combined_df


def load_user_sentiment_data(
    file_pattern: str = "user_sentiment*.csv",
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load user-provided sentiment data from CSV files.
    
    Expected format:
    - date: Date column (will be parsed)
    - ticker: Ticker symbol
    - sentiment_score: Sentiment score (ideally -1 to 1)
    - news_count: Number of articles/posts (optional)
    
    Args:
        file_pattern: Glob pattern to match CSV files
        required_columns: List of required column names
        
    Returns:
        DataFrame with loaded sentiment data
    """
    required_columns = required_columns or ["date", "ticker", "sentiment_score"]
    
    # Find all matching files
    files = list(RAW_DATA_DIR.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(
            f"No user sentiment data found matching '{file_pattern}' in {RAW_DATA_DIR}"
        )
    
    print(f"Loading user sentiment data from {len(files)} file(s)...")
    
    all_data = []
    for file in files:
        print(f"  Loading {file.name}...")
        df = pd.read_csv(file)
        
        # Validate required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"File {file.name} is missing required columns: {missing_cols}"
            )
        
        # Parse date column
        df["date"] = pd.to_datetime(df["date"])
        
        # Add news_count if missing
        if "news_count" not in df.columns:
            df["news_count"] = 1
        
        all_data.append(df)
    
    # Combine all files
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and remove duplicates
    combined_df = combined_df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    ).reset_index(drop=True)
    
    print(f"✓ Loaded {len(combined_df)} rows of user sentiment data")
    
    return combined_df


def load_user_custom_features(
    file_pattern: str = "user_features*.csv",
) -> pd.DataFrame:
    """
    Load user-provided custom features from CSV files.
    
    Expected format:
    - date: Date column (will be parsed)
    - ticker: Ticker symbol
    - ... any custom feature columns
    
    Args:
        file_pattern: Glob pattern to match CSV files
        
    Returns:
        DataFrame with loaded custom features
    """
    # Find all matching files
    files = list(RAW_DATA_DIR.glob(file_pattern))
    
    if not files:
        print(f"No custom feature files found matching '{file_pattern}'")
        return pd.DataFrame()
    
    print(f"Loading custom features from {len(files)} file(s)...")
    
    all_data = []
    for file in files:
        print(f"  Loading {file.name}...")
        df = pd.read_csv(file)
        
        # Validate date and ticker columns exist
        if "date" not in df.columns or "ticker" not in df.columns:
            raise ValueError(
                f"File {file.name} must have 'date' and 'ticker' columns"
            )
        
        # Parse date column
        df["date"] = pd.to_datetime(df["date"])
        
        all_data.append(df)
    
    # Combine all files
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and remove duplicates
    combined_df = combined_df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    ).reset_index(drop=True)
    
    print(f"✓ Loaded {len(combined_df)} rows of custom features")
    
    return combined_df


def validate_data_format(
    df: pd.DataFrame,
    data_type: str = "price",
) -> bool:
    """
    Validate that loaded data meets expected format.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('price' or 'sentiment')
        
    Returns:
        True if valid, raises ValueError if not
    """
    if data_type == "price":
        required_cols = ["date", "ticker", "close"]
        numeric_cols = ["open", "high", "low", "close", "volume"]
    elif data_type == "sentiment":
        required_cols = ["date", "ticker", "sentiment_score"]
        numeric_cols = ["sentiment_score"]
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type")
    
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")
    
    # Check for missing values
    if df[required_cols].isnull().any().any():
        raise ValueError("Required columns contain missing values")
    
    return True


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("User Data Loader - Example Usage")
    print("=" * 80)
    print("\nTo use this module, place your CSV files in:")
    print(f"  {RAW_DATA_DIR}")
    print("\nFile naming conventions:")
    print("  - user_prices*.csv for OHLCV data")
    print("  - user_sentiment*.csv for sentiment data")
    print("  - user_features*.csv for custom features")
    print("\n" + "=" * 80)
    print("Required formats:")
    print("=" * 80)
    print("\nPrice data CSV:")
    print("  date,ticker,open,high,low,close,volume")
    print("  2023-01-01,QQQ,300.0,305.0,299.0,304.0,50000000")
    print("\nSentiment data CSV:")
    print("  date,ticker,sentiment_score,news_count")
    print("  2023-01-01,QQQ,0.25,15")
    print("=" * 80)
