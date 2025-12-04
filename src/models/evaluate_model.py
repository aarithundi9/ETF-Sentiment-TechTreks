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
    mean_absolute_error,
    mean_squared_error,
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


# ==============================================================================
# Multi-Horizon Regression Evaluation
# ==============================================================================

def evaluate_multi_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_names: list = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate multi-horizon regression predictions.
    
    Calculates per-horizon metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - Directional Accuracy: % of correctly predicted direction
    
    Args:
        y_true: Actual values, shape (n_samples, n_horizons)
        y_pred: Predicted values, shape (n_samples, n_horizons)
        horizon_names: Names for each horizon (e.g., ['5d', '1m'])
        verbose: Whether to print results
        
    Returns:
        Dictionary with metrics for each horizon and aggregate metrics
    """
    if horizon_names is None:
        horizon_names = ['5d', '1m']
    
    n_horizons = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    # Handle single horizon case
    if n_horizons == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    metrics = {
        'horizons': {},
        'aggregate': {},
    }
    
    all_mae = []
    all_rmse = []
    all_dir_acc = []
    
    for i, horizon in enumerate(horizon_names[:n_horizons]):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        # Filter out NaN values
        mask = ~np.isnan(true_i) & ~np.isnan(pred_i)
        true_i = true_i[mask]
        pred_i = pred_i[mask]
        
        # Regression metrics
        mae = mean_absolute_error(true_i, pred_i)
        mse = mean_squared_error(true_i, pred_i)
        rmse = np.sqrt(mse)
        
        # Directional accuracy (correct sign prediction)
        direction_true = np.sign(true_i)
        direction_pred = np.sign(pred_i)
        dir_accuracy = np.mean(direction_true == direction_pred)
        
        # Store metrics
        metrics['horizons'][horizon] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': dir_accuracy,
            'n_samples': len(true_i),
        }
        
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_dir_acc.append(dir_accuracy)
    
    # Aggregate metrics
    metrics['aggregate'] = {
        'mean_mae': np.mean(all_mae),
        'mean_rmse': np.mean(all_rmse),
        'mean_directional_accuracy': np.mean(all_dir_acc),
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("Multi-Horizon Regression Evaluation")
        print("=" * 70)
        
        for horizon, m in metrics['horizons'].items():
            print(f"\n{horizon.upper()} Horizon ({m['n_samples']} samples):")
            print(f"  MAE:                 {m['mae']:.6f}")
            print(f"  RMSE:                {m['rmse']:.6f}")
            print(f"  Directional Acc:     {m['directional_accuracy']:.2%}")
        
        print(f"\n{'─'*40}")
        print(f"Aggregate (Average across horizons):")
        print(f"  Mean MAE:            {metrics['aggregate']['mean_mae']:.6f}")
        print(f"  Mean RMSE:           {metrics['aggregate']['mean_rmse']:.6f}")
        print(f"  Mean Directional:    {metrics['aggregate']['mean_directional_accuracy']:.2%}")
    
    return metrics


def evaluate_multi_horizon_by_ticker(
    predictions_dict: Dict[str, Dict[str, np.ndarray]],
    horizon_names: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multi-horizon predictions across multiple tickers.
    
    Args:
        predictions_dict: Dict of ticker -> {'y_true': array, 'y_pred': array}
        horizon_names: Names for each horizon
        verbose: Print results
        
    Returns:
        DataFrame with per-ticker, per-horizon metrics
    """
    if horizon_names is None:
        horizon_names = ['5d', '1m']
    
    results = []
    
    for ticker, data in predictions_dict.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        ticker_metrics = evaluate_multi_horizon(
            y_true, y_pred, horizon_names, verbose=False
        )
        
        for horizon, m in ticker_metrics['horizons'].items():
            results.append({
                'ticker': ticker,
                'horizon': horizon,
                'mae': m['mae'],
                'rmse': m['rmse'],
                'directional_accuracy': m['directional_accuracy'],
                'n_samples': m['n_samples'],
            })
    
    df = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 70)
        print("Multi-Horizon Performance by Ticker")
        print("=" * 70)
        
        # Pivot for cleaner display
        pivot_mae = df.pivot(index='ticker', columns='horizon', values='mae')
        pivot_dir = df.pivot(index='ticker', columns='horizon', values='directional_accuracy')
        
        print("\nMAE by Ticker and Horizon:")
        print(pivot_mae.round(6).to_string())
        
        print("\nDirectional Accuracy by Ticker and Horizon:")
        print((pivot_dir * 100).round(2).to_string() + " (%)")
    
    return df


def plot_multi_horizon_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_names: list = None,
    dates: pd.Series = None,
    title: str = "Multi-Horizon Predictions",
    save_path: str = None,
):
    """
    Plot actual vs predicted values for each horizon.
    
    Args:
        y_true: Actual values (n_samples, n_horizons)
        y_pred: Predicted values (n_samples, n_horizons)
        horizon_names: Names for each horizon
        dates: Optional date series for x-axis
        title: Plot title
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    if horizon_names is None:
        horizon_names = ['5d', '1m']
    
    n_horizons = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    fig, axes = plt.subplots(n_horizons, 2, figsize=(14, 5 * n_horizons))
    
    if n_horizons == 1:
        axes = axes.reshape(1, -1)
    
    for i, horizon in enumerate(horizon_names[:n_horizons]):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        # Time series plot
        ax1 = axes[i, 0]
        x_axis = dates if dates is not None else range(len(true_i))
        ax1.plot(x_axis, true_i, label='Actual', alpha=0.7)
        ax1.plot(x_axis, pred_i, label='Predicted', alpha=0.7)
        ax1.set_title(f'{horizon.upper()} Horizon - Time Series')
        ax1.set_xlabel('Date' if dates is not None else 'Sample')
        ax1.set_ylabel('Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot (actual vs predicted)
        ax2 = axes[i, 1]
        ax2.scatter(true_i, pred_i, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ax2.set_title(f'{horizon.upper()} Horizon - Actual vs Predicted')
        ax2.set_xlabel('Actual Return')
        ax2.set_ylabel('Predicted Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def create_multi_horizon_report(
    metrics: Dict[str, Any],
    ticker: str,
    save_path: str = None,
) -> str:
    """
    Create a text report of multi-horizon model performance.
    
    Args:
        metrics: Output from evaluate_multi_horizon()
        ticker: Ticker symbol
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    lines = [
        "=" * 60,
        f"Multi-Horizon Model Report: {ticker}",
        "=" * 60,
        "",
    ]
    
    for horizon, m in metrics['horizons'].items():
        lines.extend([
            f"{horizon.upper()} Horizon Performance:",
            f"  Samples:           {m['n_samples']:,}",
            f"  MAE:               {m['mae']:.6f}",
            f"  RMSE:              {m['rmse']:.6f}",
            f"  Directional Acc:   {m['directional_accuracy']:.2%}",
            "",
        ])
    
    lines.extend([
        "-" * 40,
        "Aggregate Metrics:",
        f"  Mean MAE:          {metrics['aggregate']['mean_mae']:.6f}",
        f"  Mean RMSE:         {metrics['aggregate']['mean_rmse']:.6f}",
        f"  Mean Dir. Acc:     {metrics['aggregate']['mean_directional_accuracy']:.2%}",
        "=" * 60,
    ])
    
    report = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Example classification evaluation
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
