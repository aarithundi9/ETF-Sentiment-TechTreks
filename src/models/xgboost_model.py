"""
Simple XGBoost Model for ETF Price Direction Prediction.

A straightforward classifier that predicts whether price will go up or down
in the next 5 days and 1 month. Much simpler than LSTM and more appropriate
for the available features.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/processed")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Features to use (exclude targets and date)
EXCLUDE_COLS = ['date', 'ticker', 'target', 'forward_return', 'target_5d', 'target_1m']


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
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get numeric feature columns."""
    feature_cols = [col for col in df.columns 
                    if col not in EXCLUDE_COLS 
                    and col not in ['return_5d', 'return_1m', 'target_5d', 'target_1m']
                    and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    return feature_cols


def train_xgboost_model(ticker: str = 'QQQ'):
    """
    Train XGBoost classifiers for 5-day and 1-month direction prediction.
    """
    print("="*60)
    print(f"Training XGBoost Models for {ticker}")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data(ticker)
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Get features
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")
    
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
        
        # XGBoost parameters (simple, not overfit)
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
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Baseline (always predict up)
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
        
        # Store results
        results[horizon] = {
            'accuracy': accuracy,
            'baseline': baseline_acc,
            'improvement': accuracy - baseline_acc,
            'confusion_matrix': cm,
        }
        models[horizon] = model
    
    # Feature importance (from 5d model)
    print(f"\n{'─'*60}")
    print("Top 10 Most Important Features")
    print(f"{'─'*60}")
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': models['5d'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    # Save models
    print(f"\nSaving models...")
    
    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'results': results,
        'importance': importance_df,
        'trained_at': datetime.now().isoformat(),
        'ticker': ticker,
    }
    
    model_path = MODEL_DIR / f"{ticker}_xgboost.joblib"
    joblib.dump(model_data, model_path)
    print(f"Saved to: {model_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Accuracy':<12} {'Baseline':<12} {'Improvement':<12}")
    print(f"{'─'*46}")
    for horizon, res in results.items():
        print(f"{horizon:<10} {res['accuracy']:<12.1%} {res['baseline']:<12.1%} {res['improvement']:<+12.1%}")
    
    return model_data


def load_xgboost_model(ticker: str):
    """Load a trained XGBoost model."""
    model_path = MODEL_DIR / f"{ticker}_xgboost.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def predict_with_xgboost(ticker: str, df: pd.DataFrame = None):
    """
    Make predictions using trained XGBoost model.
    
    Returns predictions for the latest data point.
    """
    model_data = load_xgboost_model(ticker)
    
    if model_data is None:
        return None
    
    if df is None:
        df = load_and_prepare_data(ticker)
    
    # Get latest row
    latest = df.iloc[-1:].copy()
    
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
            'probability': proba[1] if pred == 1 else proba[0],
            'up_probability': proba[1],
            'down_probability': proba[0],
        }
    
    return predictions


if __name__ == "__main__":
    # Train model for QQQ
    model_data = train_xgboost_model('QQQ')
    
    # Test prediction
    print("\n" + "="*60)
    print("Latest Prediction")
    print("="*60)
    
    predictions = predict_with_xgboost('QQQ')
    if predictions:
        for horizon, pred in predictions.items():
            print(f"\n{horizon.upper()} Horizon:")
            print(f"  Direction: {pred['direction']}")
            print(f"  Confidence: {pred['probability']:.1%}")
            print(f"  P(UP): {pred['up_probability']:.1%}")
            print(f"  P(DOWN): {pred['down_probability']:.1%}")
