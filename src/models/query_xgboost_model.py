"""
Query-Based XGBoost Model for ETF Price Direction Prediction.

This model is designed for the query-based workflow:
1. User enters a query
2. LLM picks an ETF and returns 4 sentiment scores
3. Model predicts 5-day and 1-month direction using:
   - 20 technical features (market context)
   - 4 query sentiment features (user input)

Total: 24 features (reduced from 48)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/processed")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Streamlined feature set (20 technical + 4 sentiment = 24 total)
TECHNICAL_FEATURES = [
    # Price (2)
    'close', 'volume',
    
    # Moving Averages (4) - key trend indicators
    'sma_5', 'sma_20', 'ema_12', 'ema_26',
    
    # Momentum (3) - trend strength
    'rsi', 'macd', 'macd_signal',
    
    # Bollinger Bands (3) - volatility context
    'bb_middle', 'bb_upper', 'bb_width',
    
    # Lagged Prices (3) - recent price memory
    'close_lag_1', 'close_lag_5', 'close_lag_10',
    
    # Lagged Volume (2) - volume trends
    'volume_lag_5', 'volume_lag_10',
    
    # Lagged RSI (2) - momentum memory  
    'rsi_lag_1', 'rsi_lag_5',
    
    # Volatility (1)
    'volatility',
]

# Query sentiment features (from LLM)
QUERY_SENTIMENT_FEATURES = [
    'query_sentiment_score',      # Overall sentiment (-1 to 1)
    'query_sentiment_positive',   # Positive probability (0 to 1)
    'query_sentiment_negative',   # Negative probability (0 to 1)
    'query_sentiment_neutral',    # Neutral probability (0 to 1)
]

# For training, we use historical sentiment as proxy
HISTORICAL_SENTIMENT_FEATURES = [
    'sentiment_score',
    'sentiment_positive', 
    'sentiment_negative',
    'sentiment_neutral',
]

ALL_FEATURES = TECHNICAL_FEATURES + QUERY_SENTIMENT_FEATURES


def load_and_prepare_data(ticker: str):
    """Load data and create target variables."""
    filepath = DATA_DIR / f"{ticker}total.csv"
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create direction targets
    df['return_5d'] = df['close'].pct_change(periods=5).shift(-5)
    df['return_1m'] = df['close'].pct_change(periods=21).shift(-21)
    
    df['target_5d'] = (df['return_5d'] > 0).astype(int)
    df['target_1m'] = (df['return_1m'] > 0).astype(int)
    
    # Add bb_width if not present (some datasets might not have it)
    if 'bb_width' not in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Add volatility if not present
    if 'volatility' not in df.columns:
        df['volatility'] = df['close'].rolling(window=20).std()
    
    # Rename historical sentiment to query sentiment for training
    # (we train on historical sentiment, but at inference we use query sentiment)
    for hist_feat, query_feat in zip(HISTORICAL_SENTIMENT_FEATURES, QUERY_SENTIMENT_FEATURES):
        if hist_feat in df.columns:
            df[query_feat] = df[hist_feat]
        else:
            # Fill with neutral sentiment if not available
            if 'positive' in query_feat:
                df[query_feat] = 0.33
            elif 'negative' in query_feat:
                df[query_feat] = 0.33
            elif 'neutral' in query_feat:
                df[query_feat] = 0.34
            else:
                df[query_feat] = 0.0
    
    return df


def get_available_features(df: pd.DataFrame) -> list:
    """Get features that exist in the dataframe."""
    available = []
    missing = []
    
    for feat in TECHNICAL_FEATURES + QUERY_SENTIMENT_FEATURES:
        if feat in df.columns:
            available.append(feat)
        else:
            missing.append(feat)
    
    if missing:
        print(f"Warning: Missing features: {missing}")
    
    return available


def train_query_model(ticker: str = 'QQQ'):
    """
    Train streamlined XGBoost model with 24 features.
    """
    print("="*60)
    print(f"Training Query-Based XGBoost Model for {ticker}")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data(ticker)
    print(f"Loaded {len(df)} rows")
    
    # Get available features
    feature_cols = get_available_features(df)
    print(f"Using {len(feature_cols)} features:")
    print(f"  Technical: {len([f for f in feature_cols if f not in QUERY_SENTIMENT_FEATURES])}")
    print(f"  Sentiment: {len([f for f in feature_cols if f in QUERY_SENTIMENT_FEATURES])}")
    
    # Drop rows with NaN targets
    df_clean = df.dropna(subset=['target_5d', 'target_1m'] + feature_cols)
    print(f"Clean rows: {len(df_clean)}")
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models = {}
    
    for horizon, target_col in [('5d', 'target_5d'), ('1m', 'target_1m')]:
        print(f"\n{'─'*60}")
        print(f"Training {horizon.upper()} Horizon Model")
        print(f"{'─'*60}")
        
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        # Calculate class balance
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        print(f"Train: {train_pos_rate:.1%} positive (up days)")
        print(f"Test:  {test_pos_rate:.1%} positive (up days)")
        
        # XGBoost parameters
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10,
        )
        
        # Train with early stopping
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        baseline_acc = test_pos_rate
        
        print(f"\nResults:")
        print(f"  Accuracy:           {accuracy:.1%}")
        print(f"  Baseline (all up):  {baseline_acc:.1%}")
        print(f"  vs Baseline:        {accuracy - baseline_acc:+.1%}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    Predicted:    DOWN    UP")
        print(f"    Actual DOWN:  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"    Actual UP:    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        results[horizon] = {
            'accuracy': accuracy,
            'baseline': baseline_acc,
            'improvement': accuracy - baseline_acc,
        }
        models[horizon] = model
    
    # Feature importance
    print(f"\n{'─'*60}")
    print("Top 10 Most Important Features")
    print(f"{'─'*60}")
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': models['5d'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        feat_type = "SENT" if row['feature'] in QUERY_SENTIMENT_FEATURES else "TECH"
        print(f"  [{feat_type}] {row['feature']:30} {row['importance']:.4f}")
    
    # Save model
    print(f"\nSaving model...")
    
    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'technical_features': [f for f in feature_cols if f not in QUERY_SENTIMENT_FEATURES],
        'sentiment_features': QUERY_SENTIMENT_FEATURES,
        'results': results,
        'importance': importance_df,
        'trained_at': datetime.now().isoformat(),
        'ticker': ticker,
        'model_type': 'query_xgboost',
    }
    
    model_path = MODEL_DIR / f"{ticker}_query_xgboost.joblib"
    joblib.dump(model_data, model_path)
    print(f"Saved to: {model_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Features: {len(feature_cols)} total ({len(model_data['technical_features'])} tech + {len(QUERY_SENTIMENT_FEATURES)} sentiment)")
    print(f"\n{'Horizon':<10} {'Accuracy':<12} {'Baseline':<12} {'Improvement':<12}")
    print(f"{'─'*46}")
    for horizon, res in results.items():
        print(f"{horizon:<10} {res['accuracy']:<12.1%} {res['baseline']:<12.1%} {res['improvement']:<+12.1%}")
    
    return model_data


def predict_with_query(ticker: str, query_sentiment: dict) -> dict:
    """
    Make predictions using query sentiment.
    
    Args:
        ticker: ETF ticker symbol
        query_sentiment: Dict with keys:
            - score: float (-1 to 1)
            - positive: float (0 to 1)
            - negative: float (0 to 1)
            - neutral: float (0 to 1)
    
    Returns:
        Predictions for 5-day and 1-month horizons
    """
    # Load model
    model_path = MODEL_DIR / f"{ticker}_query_xgboost.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    
    # Load latest market data
    df = load_and_prepare_data(ticker)
    latest = df.iloc[-1:].copy()
    
    # Override sentiment with query sentiment
    latest['query_sentiment_score'] = query_sentiment.get('score', 0)
    latest['query_sentiment_positive'] = query_sentiment.get('positive', 0.33)
    latest['query_sentiment_negative'] = query_sentiment.get('negative', 0.33)
    latest['query_sentiment_neutral'] = query_sentiment.get('neutral', 0.34)
    
    # Prepare features
    feature_cols = model_data['feature_cols']
    X = latest[feature_cols].fillna(0)
    X_scaled = model_data['scaler'].transform(X)
    
    predictions = {}
    
    for horizon in ['5d', '1m']:
        model = model_data['models'][horizon]
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        predictions[horizon] = {
            'direction': 'UP' if pred == 1 else 'DOWN',
            'confidence': max(proba),
            'up_probability': proba[1],
            'down_probability': proba[0],
        }
    
    return {
        'ticker': ticker,
        'latest_price': latest['close'].values[0],
        'latest_date': latest['date'].values[0],
        'query_sentiment': query_sentiment,
        'predictions': predictions,
    }


def train_all_etfs():
    """Train query models for all ETFs."""
    etfs = ['QQQ', 'XLY', 'XLI', 'XLF']
    
    for ticker in etfs:
        try:
            train_query_model(ticker)
        except Exception as e:
            print(f"Error training {ticker}: {e}")
        print("\n")


if __name__ == "__main__":
    # Train all ETFs
    train_all_etfs()
    
    # Test prediction with sample query sentiment
    print("\n" + "="*60)
    print("Testing Query-Based Prediction")
    print("="*60)
    
    # Simulate: "AI chip demand is surging" → Bullish sentiment
    query_sentiment = {
        'score': 0.7,        # Positive overall
        'positive': 0.75,    # 75% positive
        'negative': 0.10,    # 10% negative
        'neutral': 0.15,     # 15% neutral
    }
    
    print(f"\nQuery sentiment: {query_sentiment}")
    
    result = predict_with_query('QQQ', query_sentiment)
    if result:
        print(f"\nPrediction for {result['ticker']}:")
        print(f"  Latest Price: ${result['latest_price']:.2f}")
        for horizon, pred in result['predictions'].items():
            print(f"\n  {horizon.upper()} Horizon:")
            print(f"    Direction: {pred['direction']}")
            print(f"    Confidence: {pred['confidence']:.1%}")
            print(f"    P(UP): {pred['up_probability']:.1%}")
