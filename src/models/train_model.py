"""
Model training module.

This module handles:
- Training baseline models (logistic regression)
- Hyperparameter tuning
- Model serialization
- Cross-validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from src.config.settings import MODEL_CONFIG, DATA_DIR


class ETFPricePredictor:
    """
    Main model class for ETF price movement prediction.
    
    This is a simple baseline using logistic regression.
    Can be extended to support other models.
    """
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use
            hyperparameters: Model hyperparameters
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.hyperparameters = hyperparameters or MODEL_CONFIG["hyperparameters"]
        
        # Initialize scaler and model
        self.scaler = StandardScaler()
        self.model = self._create_model()
        
        # Training metadata
        self.is_trained = False
        self.feature_names = None
        self.training_date = None
        self.training_metrics = {}
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                **self.hyperparameters
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True,
    ) -> "ETFPricePredictor":
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            verbose: Print training progress
            
        Returns:
            Self (for method chaining)
        """
        if verbose:
            print("=" * 80)
            print(f"Training {self.model_type.upper()} Model")
            print("=" * 80)
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {len(X_train.columns)}")
            print(f"Target distribution: {y_train.value_counts(normalize=True).to_dict()}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        if verbose:
            print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        if verbose:
            print(f"Training {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Mark as trained
        self.is_trained = True
        self.training_date = datetime.now()
        
        # Calculate training accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        self.training_metrics["train_accuracy"] = train_score
        
        if verbose:
            print(f"✓ Training complete!")
            print(f"  Training accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            verbose: Print results
            
        Returns:
            Dictionary of CV metrics
        """
        if verbose:
            print(f"\nPerforming {cv}-fold cross-validation...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        cv_scores = cross_val_score(
            self.model,
            X_scaled,
            y,
            cv=cv,
            scoring='accuracy',
        )
        
        results = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }
        
        if verbose:
            print(f"✓ CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        self.training_metrics.update(results)
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (for logistic regression, use coefficients).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type == "logistic_regression":
            # For binary classification, coefficients indicate importance
            coefficients = self.model.coef_[0]
            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }).sort_values("abs_coefficient", ascending=False)
            
            return importance_df.head(top_n)
        else:
            raise NotImplementedError(f"Feature importance not implemented for {self.model_type}")
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save to (default: data/model_YYYYMMDD.pkl)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = DATA_DIR / f"model_{date_str}.pkl"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"✓ Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "ETFPricePredictor":
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded from {filepath}")
        return model


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    perform_cv: bool = True,
    save_model: bool = True,
) -> Tuple[ETFPricePredictor, Dict[str, Any]]:
    """
    Train and evaluate a baseline model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        perform_cv: Whether to perform cross-validation
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Create and train model
    model = ETFPricePredictor(
        model_type=MODEL_CONFIG["model_type"],
        hyperparameters=MODEL_CONFIG["hyperparameters"],
        random_state=MODEL_CONFIG["random_state"],
    )
    
    model.fit(X_train, y_train, verbose=True)
    
    # Cross-validation
    if perform_cv:
        model.cross_validate(
            X_train,
            y_train,
            cv=MODEL_CONFIG["cross_validation_folds"],
            verbose=True,
        )
    
    # Evaluate on test set
    from src.models.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test, verbose=True)
    
    # Show feature importance
    print("\n" + "=" * 80)
    print("Top 15 Most Important Features:")
    print("=" * 80)
    importance = model.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))
    
    # Save model
    if save_model:
        print("\n" + "=" * 80)
        model.save()
    
    return model, metrics


if __name__ == "__main__":
    # Example: Train a model on the processed data
    print("=" * 80)
    print("Training ETF Price Movement Prediction Model")
    print("=" * 80)
    
    # Load processed data
    from src.features.build_features import create_feature_pipeline, prepare_train_test_split
    
    print("\n[1/2] Running feature engineering pipeline...")
    df = create_feature_pipeline(use_mock_data=True, save_interim=False)
    
    print("\n[2/2] Preparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Train model
    print("\n")
    model, metrics = train_baseline_model(
        X_train, y_train, X_test, y_test,
        perform_cv=True,
        save_model=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ Model training complete!")
    print("=" * 80)
