"""
Model evaluation module.

This module provides comprehensive evaluation metrics and visualization
for trained models, including:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Classification report
- ROC curve and AUC (for future implementation)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test target
        verbose: Print detailed results
        
    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Model Evaluation on Test Set")
        print("=" * 80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Try to get probability predictions if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_proba = None
        has_proba = False
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "n_test_samples": len(y_test),
        "has_probability": has_proba,
    }
    
    if verbose:
        print(f"\nTest Set Size: {len(y_test)} samples")
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0, 0]:>6}  |  False Positives: {cm[0, 1]:>6}")
        print(f"  False Negatives: {cm[1, 0]:>6}  |  True Positives:  {cm[1, 1]:>6}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    return metrics


def evaluate_by_ticker(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate model performance by ticker.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        test_df: Original test DataFrame with 'ticker' column
        verbose: Print results
        
    Returns:
        DataFrame with per-ticker metrics
    """
    if 'ticker' not in test_df.columns:
        raise ValueError("test_df must contain 'ticker' column")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Align indices
    test_df_aligned = test_df.loc[X_test.index].copy()
    test_df_aligned['y_true'] = y_test.values
    test_df_aligned['y_pred'] = y_pred
    
    # Calculate metrics per ticker
    ticker_metrics = []
    for ticker in test_df_aligned['ticker'].unique():
        ticker_data = test_df_aligned[test_df_aligned['ticker'] == ticker]
        
        accuracy = accuracy_score(ticker_data['y_true'], ticker_data['y_pred'])
        precision = precision_score(ticker_data['y_true'], ticker_data['y_pred'], zero_division=0)
        recall = recall_score(ticker_data['y_true'], ticker_data['y_pred'], zero_division=0)
        f1 = f1_score(ticker_data['y_true'], ticker_data['y_pred'], zero_division=0)
        
        ticker_metrics.append({
            'ticker': ticker,
            'n_samples': len(ticker_data),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })
    
    metrics_df = pd.DataFrame(ticker_metrics)
    
    if verbose:
        print("\n" + "=" * 80)
        print("Performance by Ticker")
        print("=" * 80)
        print(metrics_df.to_string(index=False))
    
    return metrics_df


def calculate_trading_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate hypothetical trading performance.
    
    Strategy: Go long when model predicts up, stay in cash otherwise.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        test_df: Original test DataFrame with price data
        initial_capital: Starting capital
        verbose: Print results
        
    Returns:
        Dictionary with trading metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Align data
    test_df_aligned = test_df.loc[X_test.index].copy()
    test_df_aligned['y_pred'] = y_pred
    test_df_aligned['y_true'] = y_test.values
    
    # Calculate returns
    if 'forward_return' in test_df_aligned.columns:
        test_df_aligned['actual_return'] = test_df_aligned['forward_return']
    else:
        # Calculate from close prices if not available
        test_df_aligned['actual_return'] = test_df_aligned.groupby('ticker')['close'].pct_change()
    
    # Strategy: only trade when predicting up
    test_df_aligned['strategy_return'] = test_df_aligned.apply(
        lambda row: row['actual_return'] if row['y_pred'] == 1 else 0,
        axis=1
    )
    
    # Calculate cumulative returns
    cumulative_strategy = (1 + test_df_aligned['strategy_return']).cumprod()
    cumulative_buy_hold = (1 + test_df_aligned['actual_return']).cumprod()
    
    # Final values
    final_strategy_value = initial_capital * cumulative_strategy.iloc[-1]
    final_buy_hold_value = initial_capital * cumulative_buy_hold.iloc[-1]
    
    # Total returns
    strategy_return = (final_strategy_value - initial_capital) / initial_capital
    buy_hold_return = (final_buy_hold_value - initial_capital) / initial_capital
    
    # Number of trades
    n_trades = (y_pred == 1).sum()
    
    metrics = {
        'initial_capital': initial_capital,
        'final_strategy_value': final_strategy_value,
        'final_buy_hold_value': final_buy_hold_value,
        'strategy_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'n_trades': n_trades,
        'outperformance': strategy_return - buy_hold_return,
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("Hypothetical Trading Performance")
        print("=" * 80)
        print(f"Initial Capital:        ${initial_capital:,.2f}")
        print(f"\nStrategy (Model-Based):")
        print(f"  Final Value:          ${final_strategy_value:,.2f}")
        print(f"  Total Return:         {strategy_return:>7.2%}")
        print(f"  Number of Trades:     {n_trades}")
        print(f"\nBuy & Hold Benchmark:")
        print(f"  Final Value:          ${final_buy_hold_value:,.2f}")
        print(f"  Total Return:         {buy_hold_return:>7.2%}")
        print(f"\nOutperformance:         {metrics['outperformance']:>7.2%}")
        
        if strategy_return > buy_hold_return:
            print("✓ Strategy outperformed buy & hold!")
        else:
            print("⚠ Strategy underperformed buy & hold")
    
    return metrics


if __name__ == "__main__":
    # Example evaluation
    print("=" * 80)
    print("Model Evaluation Example")
    print("=" * 80)
    
    from src.features.build_features import create_feature_pipeline, prepare_train_test_split
    from src.models.train_model import ETFPricePredictor
    
    # Generate data and train model
    print("\nGenerating data and training model...")
    df = create_feature_pipeline(use_mock_data=True, save_interim=False)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    model = ETFPricePredictor()
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, verbose=True)
    
    # Evaluate by ticker (need to get test indices from df)
    test_split_idx = int(len(df) * 0.8)
    test_df = df.iloc[test_split_idx:].reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    ticker_metrics = evaluate_by_ticker(
        model, X_test_reset, y_test_reset, test_df, verbose=True
    )
    
    # Calculate trading performance
    trading_metrics = calculate_trading_performance(
        model, X_test_reset, y_test_reset, test_df, verbose=True
    )
