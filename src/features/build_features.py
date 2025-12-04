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


# ==============================================================================
# Multi-Horizon Target Construction
# ==============================================================================

def make_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: dict = None,
    target_type: str = "return",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Create multi-horizon target columns for price prediction.
    
    Args:
        df: DataFrame with price data (must have 'close' column and be sorted by date)
        horizons: Dict mapping horizon name to days, e.g., {"5d": 5, "1m": 21}
                  If None, uses MULTI_HORIZON_CONFIG["horizons"]
        target_type: 'return' (percentage change) or 'price_change' (absolute)
        price_col: Column name for price data
        
    Returns:
        DataFrame with added target columns:
        - target_5d: 5-day ahead return/change
        - target_1m: 1-month ahead return/change
        
    Example:
        >>> df = make_multi_horizon_targets(df, horizons={"5d": 5, "1m": 21})
        >>> # Now df has 'target_5d' and 'target_1m' columns
    """
    from src.config.settings import MULTI_HORIZON_CONFIG
    
    df = df.copy()
    
    # Use default horizons if not provided
    if horizons is None:
        horizons = MULTI_HORIZON_CONFIG["horizons"]
    
    if target_type is None:
        target_type = MULTI_HORIZON_CONFIG.get("target_type", "return")
    
    # Ensure sorted by date within each ticker
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)
    
    for horizon_name, horizon_days in horizons.items():
        target_col = f"target_{horizon_name}"
        
        if "ticker" in df.columns:
            # Multi-ticker: compute per ticker
            if target_type == "return":
                # Percentage return: (future_price - current_price) / current_price
                df[target_col] = df.groupby("ticker")[price_col].transform(
                    lambda x: x.pct_change(periods=horizon_days).shift(-horizon_days)
                )
            else:  # price_change
                # Absolute price change: future_price - current_price
                df[target_col] = df.groupby("ticker")[price_col].transform(
                    lambda x: x.diff(periods=horizon_days).shift(-horizon_days)
                )
        else:
            # Single ticker
            if target_type == "return":
                df[target_col] = df[price_col].pct_change(periods=horizon_days).shift(-horizon_days)
            else:
                df[target_col] = df[price_col].diff(periods=horizon_days).shift(-horizon_days)
    
    return df


def prepare_multi_horizon_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    sequence_length: int = 20,
    split_ratio: float = 0.8,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequential data for LSTM/temporal models with multi-horizon targets.
    
    Creates sequences of features for each prediction, respecting time ordering.
    
    Args:
        df: DataFrame with features and multi-horizon targets
        feature_cols: List of feature column names
        target_cols: List of target column names (e.g., ['target_5d', 'target_1m'])
        sequence_length: Number of time steps in each input sequence
        split_ratio: Train/test split ratio
        val_ratio: Validation ratio (from training portion)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        - X arrays have shape (n_samples, sequence_length, n_features)
        - y arrays have shape (n_samples, n_horizons)
    """
    # Ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Drop rows with NaN in features or targets
    cols_to_check = feature_cols + target_cols
    df = df.dropna(subset=[col for col in cols_to_check if col in df.columns])
    
    # Extract feature and target arrays
    X_data = df[feature_cols].values
    y_data = df[target_cols].values
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(df) - sequence_length):
        X_sequences.append(X_data[i:i + sequence_length])
        y_sequences.append(y_data[i + sequence_length - 1])  # Target at end of sequence
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    # Time-based split
    total_samples = len(X)
    train_end = int(total_samples * split_ratio * (1 - val_ratio))
    val_end = int(total_samples * split_ratio)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"Sequence preparation complete:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Horizons: {len(target_cols)}")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Val:   {len(X_val)} sequences")
    print(f"  Test:  {len(X_test)} sequences")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_ticker_dataset(ticker: str) -> pd.DataFrame:
    """
    Load a processed ticker dataset from the data/processed directory.
    
    Args:
        ticker: Ticker symbol (e.g., 'QQQ', 'XLY')
        
    Returns:
        DataFrame with all features and original targets
    """
    filepath = PROCESSED_DATA_DIR / f"{ticker}total.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {ticker} dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


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
