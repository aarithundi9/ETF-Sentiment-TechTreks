"""
Feature engineering module.

This module combines technical indicators and sentiment data to create
features for machine learning models. It handles:
- Merging price and sentiment data
- Creating lagged features
- Creating target variable (price direction)
- Train/test splitting
- Saving processed datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    FEATURE_CONFIG,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
)
from src.data.technical_data import TechnicalDataFetcher, add_all_technical_indicators
from src.data.sentiment_data import SentimentDataFetcher, calculate_sentiment_features


def merge_price_and_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and sentiment data on date and ticker.
    
    Args:
        price_df: DataFrame with price and technical indicators
        sentiment_df: DataFrame with sentiment scores
        
    Returns:
        Merged DataFrame
    """
    # Ensure both are sorted
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    sentiment_df = sentiment_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Merge on date and ticker
    merged = pd.merge(
        price_df,
        sentiment_df,
        on=["date", "ticker"],
        how="left",  # Keep all price data, fill missing sentiment
    )
    
    # Forward fill missing sentiment (use last known sentiment)
    sentiment_cols = [col for col in sentiment_df.columns if col not in ["date", "ticker"]]
    merged[sentiment_cols] = merged.groupby("ticker")[sentiment_cols].fillna(method="ffill")
    
    # Backward fill any remaining (at the start)
    merged[sentiment_cols] = merged.groupby("ticker")[sentiment_cols].fillna(method="bfill")
    
    # Fill any still missing with 0 (neutral sentiment)
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)
    
    return merged


def create_lagged_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lagged versions of features.
    
    Args:
        df: DataFrame with features
        feature_columns: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with added lagged features
    """
    df = df.copy()
    
    for col in feature_columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    return df


def create_target_variable(
    df: pd.DataFrame,
    forward_period: int = 1,
    target_type: str = "binary",
) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with price data
        forward_period: Days forward to calculate target
        target_type: 'binary' (up/down) or 'continuous' (return)
        
    Returns:
        DataFrame with target column added
    """
    df = df.copy()
    
    # Calculate forward return
    df["forward_return"] = df.groupby("ticker")["close"].transform(
        lambda x: x.pct_change(periods=forward_period).shift(-forward_period)
    )
    
    if target_type == "binary":
        # Binary: 1 if price goes up, 0 if down
        df["target"] = (df["forward_return"] > 0).astype(int)
    elif target_type == "continuous":
        # Continuous: the actual return
        df["target"] = df["forward_return"]
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    return df


def create_feature_pipeline(
    use_mock_data: bool = True,
    save_interim: bool = True,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Steps:
    1. Load/generate price data
    2. Load/generate sentiment data
    3. Add technical indicators
    4. Add sentiment features
    5. Merge price and sentiment
    6. Create lagged features
    7. Create target variable
    8. Save interim data
    
    Args:
        use_mock_data: If True, use mock data; otherwise use real data sources
        save_interim: If True, save intermediate datasets
        
    Returns:
        DataFrame ready for modeling
    """
    print("=" * 80)
    print("Running Feature Engineering Pipeline")
    print("=" * 80)
    
    # Step 1: Load price data
    print("\n[1/7] Loading price data...")
    source = "mock" if use_mock_data else "real"
    price_fetcher = TechnicalDataFetcher(source=source)
    price_df = price_fetcher.fetch_ohlcv()
    print(f"✓ Loaded {len(price_df)} price records")
    
    # Step 2: Load sentiment data
    print("\n[2/7] Loading sentiment data...")
    sentiment_fetcher = SentimentDataFetcher(source=source)
    sentiment_df = sentiment_fetcher.fetch_sentiment()
    print(f"✓ Loaded {len(sentiment_df)} sentiment records")
    
    # Step 3: Add technical indicators
    print("\n[3/7] Calculating technical indicators...")
    price_df = add_all_technical_indicators(price_df)
    
    if save_interim:
        interim_path = INTERIM_DATA_DIR / "prices_with_technicals.csv"
        price_df.to_csv(interim_path, index=False)
        print(f"✓ Saved to {interim_path}")
    
    # Step 4: Add sentiment features
    print("\n[4/7] Calculating sentiment features...")
    sentiment_df = calculate_sentiment_features(sentiment_df)
    
    if save_interim:
        interim_path = INTERIM_DATA_DIR / "sentiment_with_features.csv"
        sentiment_df.to_csv(interim_path, index=False)
        print(f"✓ Saved to {interim_path}")
    
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
    print(f"✓ Created lagged features for {len(lag_features)} variables")
    
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
    
    # Save final processed data
    if save_interim:
        processed_path = PROCESSED_DATA_DIR / "modeling_dataset.csv"
        merged_df.to_csv(processed_path, index=False)
        print(f"\n✓ Final dataset saved to {processed_path}")
        print(f"  Shape: {merged_df.shape}")
    
    print("\n" + "=" * 80)
    print("✓ Feature engineering pipeline complete!")
    print("=" * 80)
    
    return merged_df


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: Optional[List[str]] = None,
    split_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/test split for modeling.
    
    Uses time-based split (not random) to avoid lookahead bias.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        drop_cols: Additional columns to drop
        split_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Sort by date to ensure time-based split
    df = df.sort_values("date").reset_index(drop=True)
    
    # Columns to drop (non-feature columns)
    default_drop = ["date", "ticker", "forward_return", target_col]
    drop_cols = drop_cols or []
    all_drop_cols = list(set(default_drop + drop_cols))
    
    # Separate features and target
    X = df.drop(columns=[col for col in all_drop_cols if col in df.columns])
    y = df[target_col]
    
    # Time-based split
    split_idx = int(len(df) * split_ratio)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Remove any remaining NaN values
    # For training set, drop rows with NaN
    train_mask = ~X_train.isnull().any(axis=1)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    # For test set, fill NaN with column means from training
    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(X_train[col].mean())
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(X_train.columns)}")
    print(f"Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run the feature engineering pipeline
    df = create_feature_pipeline(use_mock_data=True, save_interim=True)
    
    # Prepare train/test split
    print("\n" + "=" * 80)
    print("Preparing Train/Test Split")
    print("=" * 80)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    print("\n" + "=" * 80)
    print("Sample Features:")
    print("=" * 80)
    print(X_train.head())
