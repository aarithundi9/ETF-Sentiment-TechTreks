"""
Global configuration settings for the ETF Sentiment Analysis project.

This module provides centralized configuration for:
- Tickers to analyze
- Date ranges
- File paths
- Database connections
- Model parameters
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# ==============================================================================
# ETF Tickers Configuration
# ==============================================================================
TICKERS: List[str] = ["QQQ"]  # Focus on QQQ for real data collection

# ==============================================================================
# Date Range Configuration
# ==============================================================================
START_DATE: str = "2015-01-01"  # Extended range for more training data
END_DATE: str = "2024-12-31"  # Up to present

# ==============================================================================
# Directory Paths
# ==============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Database Configuration
# ==============================================================================
DB_PATH = DATA_DIR / "etf_sentiment.db"

# ==============================================================================
# Mock Data Configuration
# ==============================================================================
MOCK_DATA_CONFIG = {
    "num_days": 730,  # ~2 years of data
    "initial_price": {
        "QQQ": 300.0,
        "SPY": 400.0,
        "IWM": 200.0,
    },
    "volatility": 0.02,  # Daily volatility
    "drift": 0.0003,  # Slight upward drift
}

# ==============================================================================
# Technical Indicators Configuration
# ==============================================================================
TECHNICAL_INDICATORS = {
    "sma_periods": [5, 10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
}

# ==============================================================================
# Sentiment Analysis Configuration
# ==============================================================================
SENTIMENT_CONFIG = {
    "sources": ["news", "reddit", "twitter"],  # Future data sources
    "lookback_days": 7,  # Days to aggregate sentiment
    "min_news_count": 3,  # Minimum news articles for valid sentiment
}

# ==============================================================================
# Feature Engineering Configuration
# ==============================================================================
FEATURE_CONFIG = {
    "target_column": "target",  # Binary: 1 if price up, 0 otherwise
    "target_type": "binary",  # "binary" for classification (up/down) or "continuous" for regression (actual return %)
    "lookback_periods": [1, 3, 5, 10],  # Periods for lagged features
    "forward_period": 5,  # Days forward for target calculation (5 = ~1 week)
    "train_test_split": 0.8,  # 80% train, 20% test
}

# ==============================================================================
# Model Configuration
# ==============================================================================
MODEL_CONFIG = {
    "model_type": "logistic_regression",
    "random_state": 42,
    "hyperparameters": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "cross_validation_folds": 5,
}

# ==============================================================================
# API Keys (from environment variables)
# ==============================================================================
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ETF_Sentiment_Bot/1.0")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# GDELT API Configuration
GDELT_API_KEY = os.getenv("GDELT_API_KEY", "")  # Optional - GDELT is free but rate-limited
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# ==============================================================================
# Logging Configuration
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration as a dictionary.
    
    Returns:
        Dict containing all configuration settings
    """
    return {
        "tickers": TICKERS,
        "date_range": {"start": START_DATE, "end": END_DATE},
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "raw_data": str(RAW_DATA_DIR),
            "interim_data": str(INTERIM_DATA_DIR),
            "processed_data": str(PROCESSED_DATA_DIR),
            "database": str(DB_PATH),
        },
        "mock_data": MOCK_DATA_CONFIG,
        "technical_indicators": TECHNICAL_INDICATORS,
        "sentiment": SENTIMENT_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
    }


def print_config() -> None:
    """Print current configuration settings."""
    config = get_config()
    print("=" * 80)
    print("ETF Sentiment Analysis - Configuration")
    print("=" * 80)
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
