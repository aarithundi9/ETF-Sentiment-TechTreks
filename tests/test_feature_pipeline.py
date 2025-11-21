"""
Tests for feature engineering pipeline.

Tests cover:
- Feature merging
- Lagged feature creation
- Target variable creation
- Train/test splitting
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.build_features import (
    merge_price_and_sentiment,
    create_lagged_features,
    create_target_variable,
    prepare_train_test_split,
)


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.bdate_range(start='2023-01-01', periods=50)
        return pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'close': 100 + np.cumsum(np.random.randn(50)),
            'volume': 1000000 + np.random.randint(-100000, 100000, 50),
        })
    
    @pytest.fixture
    def sample_sentiment_data(self):
        """Create sample sentiment data."""
        dates = pd.bdate_range(start='2023-01-01', periods=50)
        return pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'sentiment_score': np.random.randn(50) * 0.3,
            'news_count': np.random.randint(1, 20, 50),
        })
    
    def test_merge_price_and_sentiment(self, sample_price_data, sample_sentiment_data):
        """Test merging price and sentiment data."""
        merged = merge_price_and_sentiment(sample_price_data, sample_sentiment_data)
        
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) == len(sample_price_data)
        assert 'close' in merged.columns
        assert 'sentiment_score' in merged.columns
        # Should not have NaN in sentiment after forward/backward fill
        assert not merged['sentiment_score'].isna().any()
    
    def test_create_lagged_features(self, sample_price_data):
        """Test lagged feature creation."""
        df = create_lagged_features(
            sample_price_data,
            feature_columns=['close', 'volume'],
            lags=[1, 3, 5],
        )
        
        # Check that lagged columns were created
        assert 'close_lag_1' in df.columns
        assert 'close_lag_3' in df.columns
        assert 'volume_lag_5' in df.columns
        
        # First lag should be close to actual (just shifted)
        assert df['close_lag_1'].iloc[1] == df['close'].iloc[0]
    
    def test_create_target_variable_binary(self, sample_price_data):
        """Test binary target creation."""
        df = create_target_variable(
            sample_price_data,
            forward_period=1,
            target_type='binary',
        )
        
        assert 'target' in df.columns
        assert 'forward_return' in df.columns
        # Binary target should only be 0 or 1
        assert set(df['target'].dropna().unique()).issubset({0, 1})
    
    def test_create_target_variable_continuous(self, sample_price_data):
        """Test continuous target creation."""
        df = create_target_variable(
            sample_price_data,
            forward_period=1,
            target_type='continuous',
        )
        
        assert 'target' in df.columns
        # Continuous target should equal forward return
        assert (df['target'] == df['forward_return']).all()
    
    def test_prepare_train_test_split(self):
        """Test train/test split preparation."""
        # Create sample data
        dates = pd.bdate_range(start='2023-01-01', periods=100)
        df = pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'close': 100 + np.cumsum(np.random.randn(100)),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100),
        })
        
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df,
            target_col='target',
            split_ratio=0.8,
        )
        
        # Check shapes
        assert len(X_train) + len(X_test) <= len(df)  # May be less due to NaN removal
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that 'date' and 'target' are not in features
        assert 'date' not in X_train.columns
        assert 'target' not in X_train.columns
        
        # Check time-based split (train comes before test)
        train_dates = df.iloc[:len(X_train)]['date']
        test_dates = df.iloc[-len(X_test):]['date']
        assert train_dates.max() <= test_dates.min()


class TestFeaturePipeline:
    """Integration tests for the complete feature pipeline."""
    
    def test_feature_pipeline_runs(self):
        """Test that the complete pipeline runs without errors."""
        from src.features.build_features import create_feature_pipeline
        
        # Run pipeline (should not raise any errors)
        df = create_feature_pipeline(use_mock_data=True, save_interim=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'target' in df.columns
        
        # Should have technical indicators
        assert any('sma' in col for col in df.columns)
        assert any('ema' in col for col in df.columns)
        
        # Should have sentiment features
        assert 'sentiment_score' in df.columns
    
    def test_no_data_leakage(self):
        """Test that there's no data leakage in train/test split."""
        from src.features.build_features import create_feature_pipeline, prepare_train_test_split
        
        df = create_feature_pipeline(use_mock_data=True, save_interim=False)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df, split_ratio=0.8)
        
        # Indices should not overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
