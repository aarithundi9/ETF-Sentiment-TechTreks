"""
Tests for data pipeline components.

Tests cover:
- Mock data generation
- Technical data fetching
- Sentiment data fetching
- Data validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.mock_data_generator import (
    generate_price_series,
    generate_sentiment_series,
    generate_all_mock_data,
)
from src.data.technical_data import (
    TechnicalDataFetcher,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    add_all_technical_indicators,
)
from src.data.sentiment_data import (
    SentimentDataFetcher,
    calculate_sentiment_features,
)
from src.config.settings import TICKERS


class TestMockDataGenerator:
    """Tests for mock data generation."""
    
    def test_generate_price_series(self):
        """Test price series generation."""
        df = generate_price_series(
            ticker="TEST",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_price=100.0,
            seed=42,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert set(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']).issubset(df.columns)
        assert df['ticker'].unique()[0] == "TEST"
        assert all(df['high'] >= df['low'])
        assert all(df['high'] >= df['close'])
        assert all(df['low'] <= df['close'])
    
    def test_generate_sentiment_series(self):
        """Test sentiment series generation."""
        # First generate price data
        price_df = generate_price_series(
            ticker="TEST",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_price=100.0,
        )
        
        sentiment_df = generate_sentiment_series(
            ticker="TEST",
            price_df=price_df,
            seed=42,
        )
        
        assert isinstance(sentiment_df, pd.DataFrame)
        assert len(sentiment_df) == len(price_df)
        assert set(['date', 'ticker', 'sentiment_score', 'news_count']).issubset(sentiment_df.columns)
        assert all(sentiment_df['sentiment_score'] >= -1)
        assert all(sentiment_df['sentiment_score'] <= 1)
        assert all(sentiment_df['news_count'] >= 0)
    
    def test_generate_all_mock_data(self):
        """Test complete mock data generation."""
        prices_df, sentiment_df = generate_all_mock_data(save=False, verbose=False)
        
        assert isinstance(prices_df, pd.DataFrame)
        assert isinstance(sentiment_df, pd.DataFrame)
        assert len(prices_df) > 0
        assert len(sentiment_df) > 0
        
        # Check all tickers are present
        for ticker in TICKERS:
            assert ticker in prices_df['ticker'].values
            assert ticker in sentiment_df['ticker'].values


class TestTechnicalData:
    """Tests for technical data processing."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.bdate_range(start='2023-01-01', periods=100)
        return pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'open': 100 + np.random.randn(100),
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
            'volume': 1000000 + np.random.randint(-100000, 100000, 100),
        })
    
    def test_technical_data_fetcher(self):
        """Test technical data fetcher."""
        fetcher = TechnicalDataFetcher(source="mock")
        df = fetcher.fetch_ohlcv()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'close' in df.columns
    
    def test_calculate_sma(self, sample_price_data):
        """Test SMA calculation."""
        df = calculate_sma(sample_price_data, periods=[5, 10])
        
        assert 'sma_5' in df.columns
        assert 'sma_10' in df.columns
        assert not df['sma_5'].isna().all()
    
    def test_calculate_ema(self, sample_price_data):
        """Test EMA calculation."""
        df = calculate_ema(sample_price_data, periods=[12, 26])
        
        assert 'ema_12' in df.columns
        assert 'ema_26' in df.columns
        assert not df['ema_12'].isna().all()
    
    def test_calculate_rsi(self, sample_price_data):
        """Test RSI calculation."""
        df = calculate_rsi(sample_price_data, period=14)
        
        assert 'rsi' in df.columns
        # RSI should be between 0 and 100
        assert df['rsi'].dropna().between(0, 100).all()
    
    def test_add_all_technical_indicators(self, sample_price_data):
        """Test adding all technical indicators."""
        df = add_all_technical_indicators(sample_price_data)
        
        # Check that various indicators were added
        expected_cols = ['sma_5', 'ema_12', 'rsi', 'macd', 'bb_upper']
        for col in expected_cols:
            assert col in df.columns


class TestSentimentData:
    """Tests for sentiment data processing."""
    
    def test_sentiment_data_fetcher(self):
        """Test sentiment data fetcher."""
        fetcher = SentimentDataFetcher(source="mock")
        df = fetcher.fetch_sentiment()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sentiment_score' in df.columns
    
    def test_calculate_sentiment_features(self):
        """Test sentiment feature calculation."""
        # Create sample sentiment data
        dates = pd.bdate_range(start='2023-01-01', periods=50)
        sentiment_df = pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'sentiment_score': np.random.randn(50) * 0.3,
            'news_count': np.random.randint(1, 20, 50),
        })
        
        df = calculate_sentiment_features(sentiment_df)
        
        # Check that features were added
        assert 'sentiment_ma_7' in df.columns
        assert 'sentiment_change' in df.columns
        assert 'sentiment_volatility' in df.columns


class TestDataValidation:
    """Tests for data validation."""
    
    def test_price_data_completeness(self):
        """Test that price data has no missing required fields."""
        fetcher = TechnicalDataFetcher(source="mock")
        df = fetcher.fetch_ohlcv()
        
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in df.columns
            assert not df[col].isna().all()
    
    def test_sentiment_data_range(self):
        """Test that sentiment scores are in valid range."""
        fetcher = SentimentDataFetcher(source="mock")
        df = fetcher.fetch_sentiment()
        
        assert df['sentiment_score'].between(-1, 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
