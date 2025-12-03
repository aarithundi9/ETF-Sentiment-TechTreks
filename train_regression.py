"""
Feature Selection and Regression Model Training

This script:
1. Uses logistic regression to identify important features
2. Selects top N features based on importance
3. Trains regression models (Linear, Random Forest, XGBoost) for price prediction
4. Compares performance (RMSE, MAE, R²)
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.features.build_features import create_feature_pipeline, prepare_train_test_split
from src.config.settings import FEATURE_CONFIG
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def get_top_features_from_logistic(model_path: str, top_n: int = 15) -> list:
    """
    Load logistic regression model and extract top N features by importance.
    
    Args:
        model_path: Path to saved logistic regression model
        top_n: Number of top features to select
        
    Returns:
        List of top feature names
    """
    print(f"\n[1/5] Loading logistic regression model for feature selection...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get feature importance (absolute coefficient values)
    coef = model.model.coef_[0]
    feature_names = model.feature_names
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coef)
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    print(f"✓ Selected top {top_n} features based on logistic regression importance:")
    for i, feat in enumerate(top_features, 1):
        imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        print(f"   {i:2d}. {feat:25s} (importance: {imp:.4f})")
    
    return top_features


def create_regression_dataset(use_mock_data: bool = False) -> pd.DataFrame:
    """
    Create dataset with continuous target (price return).
    
    Args:
        use_mock_data: If True, use mock data; otherwise use real data
        
    Returns:
        DataFrame with features and continuous target
    """
    print(f"\n[2/5] Creating regression dataset with continuous target...")
    
    # Import here to avoid circular imports
    from src.data.technical_data import TechnicalDataFetcher, add_all_technical_indicators
    from src.data.sentiment_data import SentimentDataFetcher, calculate_sentiment_features
    from src.features.build_features import merge_price_and_sentiment, create_lagged_features, create_target_variable
    
    # Load data
    source = "mock" if use_mock_data else "real"
    
    price_fetcher = TechnicalDataFetcher(source=source)
    price_df = price_fetcher.fetch_ohlcv()
    price_df = add_all_technical_indicators(price_df)
    
    sentiment_fetcher = SentimentDataFetcher(source=source)
    sentiment_df = sentiment_fetcher.fetch_sentiment()
    sentiment_df = calculate_sentiment_features(sentiment_df)
    
    # Merge
    df = merge_price_and_sentiment(price_df, sentiment_df)
    
    # Create lagged features
    lag_features = ['close', 'volume', 'sentiment_score', 'rsi']
    df = create_lagged_features(df, lag_features, lags=[1, 3, 5, 10])
    
    # Create CONTINUOUS target
    df = create_target_variable(
        df,
        forward_period=FEATURE_CONFIG['forward_period'],
        target_type='continuous'  # Force continuous
    )
    
    # Drop rows with NaN in target
    before_drop = len(df)
    df = df.dropna(subset=['target'])
    after_drop = len(df)
    
    if before_drop > after_drop:
        print(f"  ⚠ Dropped {before_drop - after_drop} rows with missing target")
    
    print(f"✓ Created dataset with {len(df)} samples")
    print(f"  Target range: {df['target'].min():.4f} to {df['target'].max():.4f}")
    print(f"  Target mean: {df['target'].mean():.4f}, std: {df['target'].std():.4f}")
    
    return df


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train multiple regression models and compare performance.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with trained models and metrics
    """
    print(f"\n[4/5] Training regression models...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (L2 Regularization)': Ridge(alpha=1.0),
        'Lasso (L1 Regularization)': Lasso(alpha=0.01, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'predictions': y_pred_test,
        }
        
        print(f"    Train RMSE: {train_rmse:.4f}")
        print(f"    Test RMSE:  {test_rmse:.4f}")
        print(f"    Test MAE:   {test_mae:.4f}")
        print(f"    Test R²:    {test_r2:.4f}")
    
    return results


def display_results(results: dict, y_test: pd.Series):
    """Display comparison of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - PRICE PREDICTION (Continuous)")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Model': results.keys(),
        'RMSE': [r['test_rmse'] for r in results.values()],
        'MAE': [r['test_mae'] for r in results.values()],
        'R²': [r['test_r2'] for r in results.values()],
    }).sort_values('RMSE')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_results = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   RMSE: {best_results['test_rmse']:.4f}")
    print(f"   MAE:  {best_results['test_mae']:.4f}")
    print(f"   R²:   {best_results['test_r2']:.4f}")
    
    print("\n💡 Interpretation:")
    print(f"   • RMSE: Average prediction error = {best_results['test_rmse']:.2%} return")
    print(f"   • MAE:  Typical error = {best_results['test_mae']:.2%} return")
    print(f"   • R²:   Model explains {best_results['test_r2']:.1%} of variance")
    
    # Show some example predictions
    print(f"\n📊 Sample Predictions (first 10 test samples):")
    print(f"{'Actual Return':<15} {'Predicted Return':<18} {'Error':<10}")
    print("-" * 45)
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        pred = best_results['predictions'][i]
        error = actual - pred
        print(f"{actual:>13.2%}  {pred:>16.2%}  {error:>8.2%}")
    
    return best_model_name, best_results


def save_regression_model(model_name: str, model_data: dict, feature_names: list):
    """Save the best regression model."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(f"data/regression_model_{timestamp}.pkl")
    
    save_data = {
        'model_name': model_name,
        'model': model_data['model'],
        'scaler': model_data['scaler'],
        'feature_names': feature_names,
        'metrics': {
            'test_rmse': model_data['test_rmse'],
            'test_mae': model_data['test_mae'],
            'test_r2': model_data['test_r2'],
        },
        'timestamp': timestamp,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    return model_path


def main(use_mock_data: bool = False, top_n_features: int = 15):
    """
    Main workflow for feature selection and regression.
    
    Args:
        use_mock_data: If True, use mock data; otherwise use real data
        top_n_features: Number of top features to select from logistic regression
    """
    print("=" * 80)
    print("FEATURE SELECTION & REGRESSION MODEL TRAINING")
    print("=" * 80)
    
    # Step 1: Get top features from logistic regression
    import glob
    logistic_models = sorted(glob.glob('data/model_*.pkl'))
    if not logistic_models:
        print("\n❌ No logistic regression model found!")
        print("   Run: python main.py train --use-real-data")
        return
    
    latest_logistic = logistic_models[-1]
    top_features = get_top_features_from_logistic(latest_logistic, top_n=top_n_features)
    
    # Step 2: Create regression dataset (continuous target)
    df = create_regression_dataset(use_mock_data=use_mock_data)
    
    # Step 3: Filter to top features
    print(f"\n[3/5] Filtering dataset to top {top_n_features} features...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['date', 'ticker', 'target', 'forward_return']]
    available_features = [f for f in top_features if f in feature_cols]
    
    print(f"✓ Using {len(available_features)} features (some may not be in dataset)")
    
    X = df[available_features]
    y = df['target']
    
    # Drop any remaining NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"✓ After removing NaN: {len(X)} samples remain")
    
    # Time-based split
    split_idx = int(len(X) * FEATURE_CONFIG['train_test_split'])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Step 4: Train regression models
    results = train_regression_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Display results and save best model
    print(f"\n[5/5] Evaluating and saving best model...")
    best_name, best_results = display_results(results, y_test)
    model_path = save_regression_model(best_name, best_results, available_features)
    
    print("\n" + "=" * 80)
    print("✓ FEATURE SELECTION & REGRESSION TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review feature importance and model performance")
    print(f"  2. Try different top_n_features values (e.g., 10, 20, 30)")
    print(f"  3. Experiment with hyperparameter tuning")
    print(f"  4. Use saved model for price predictions")
    print(f"\nTo use the saved model:")
    print(f"  import pickle")
    print(f"  with open('{model_path}', 'rb') as f:")
    print(f"      model_data = pickle.load(f)")
    print(f"  predictions = model_data['model'].predict(model_data['scaler'].transform(X_new))")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature selection and regression training")
    parser.add_argument('--use-mock-data', action='store_true', help='Use mock data instead of real data')
    parser.add_argument('--top-features', type=int, default=15, help='Number of top features to select (default: 15)')
    
    args = parser.parse_args()
    
    main(use_mock_data=args.use_mock_data, top_n_features=args.top_features)
