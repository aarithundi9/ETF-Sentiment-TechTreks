# Regression Model Training - Quick Guide

## What You Have Now

✅ **Classification Model** (Logistic Regression)
- Predicts: UP (1) or DOWN (0)
- Output: Binary class
- File: `data/model_YYYYMMDD_HHMMSS.pkl`
- Test Accuracy: ~56%

✅ **Regression Model** (Lasso/Ridge/Random Forest/etc.)
- Predicts: Actual return percentage
- Output: Continuous value (e.g., +2.5%, -1.3%)
- File: `data/regression_model_YYYYMMDD_HHMMSS.pkl`
- Test RMSE: ~2.5% error

## Your Workflow (Feature Selection → Regression)

### Step 1: Train Classification Model (Feature Selection)
```bash
python main.py train --use-real-data
```
This trains logistic regression and identifies important features.

### Step 2: Train Regression Model with Best Features
```bash
# Use top 15 features from logistic regression
python train_regression.py --top-features 15

# Or try different amounts
python train_regression.py --top-features 10   # Fewer features
python train_regression.py --top-features 20   # More features
```

## What Models Were Tested

The script automatically trains and compares:

1. **Linear Regression** - Simple baseline
2. **Ridge Regression** - L2 regularization (prevents overfitting)
3. **Lasso Regression** - L1 regularization (feature selection)
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Boosted trees

Best model is automatically selected and saved.

## Understanding the Output

**RMSE (Root Mean Squared Error):** 0.0251 = 2.51%
- Average prediction error
- Lower is better
- Your model's predictions are off by ~2.5% on average

**MAE (Mean Absolute Error):** 0.0200 = 2.00%
- Typical error (less sensitive to outliers)
- Lower is better
- Typical prediction is off by ~2%

**R² (R-squared):** -0.0308 = -3.1%
- How much variance the model explains
- Range: -∞ to 1.0 (1.0 = perfect)
- Negative = model worse than just predicting the mean
- **Your model needs improvement!**

## Why R² is Negative

Your model is essentially predicting ~0.30% return for everything, which is just the average. This means:
- Features aren't predictive enough
- Model is too simple
- Need more/better features OR different approach

## How to Improve

### 1. Try More Features
```bash
python train_regression.py --top-features 30
```

### 2. Add New Features
Edit `src/features/build_features.py` and add:
- Volume indicators (OBV, VWAP)
- More technical indicators
- Market sentiment (VIX, put/call ratio)
- Sector performance
- Macro indicators (interest rates, unemployment)

### 3. Try Different Models
The script already tests multiple models. Random Forest and Gradient Boosting can capture non-linear patterns.

### 4. Hyperparameter Tuning
Edit `train_regression.py` to tune:
- `Ridge(alpha=...)` - Try 0.1, 1.0, 10.0
- `RandomForest(n_estimators=..., max_depth=...)` - More trees, deeper trees
- `GradientBoosting(learning_rate=..., n_estimators=...)`

### 5. Feature Engineering
Instead of raw features, create:
- Feature ratios (e.g., close / sma_50)
- Feature differences (e.g., rsi - rsi_lag_1)
- Interaction terms (e.g., sentiment * volatility)

## Using the Saved Model

### Load Model
```python
import pickle
import pandas as pd

# Load
with open('data/regression_model_20251201_195918.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Components
model = model_data['model']           # Trained Lasso
scaler = model_data['scaler']         # StandardScaler
features = model_data['feature_names'] # List of 15 features
metrics = model_data['metrics']        # RMSE, MAE, R²
```

### Make Predictions
```python
# Prepare new data with same 15 features
new_data = pd.DataFrame({
    'bb_upper': [200.5],
    'bb_lower': [195.2],
    'close': [197.8],
    # ... all 15 features
})

# Scale and predict
scaled_data = scaler.transform(new_data[features])
predicted_return = model.predict(scaled_data)

print(f"Predicted 5-day return: {predicted_return[0]:.2%}")
# Example: "Predicted 5-day return: +2.34%"
```

## File Locations

```
data/
├── model_*.pkl                      # Classification models (UP/DOWN)
├── regression_model_*.pkl           # Regression models (price prediction)
├── processed/modeling_dataset.csv   # Combined dataset
└── model_weights.csv                # Feature importance from logistic regression
```

## Next Steps

1. ✅ You have the framework working
2. ⚠️ Model performance needs improvement (R² negative)
3. 🔄 Try these in order:
   - Add more features (--top-features 30)
   - Engineer better features (ratios, interactions)
   - Tune hyperparameters
   - Try neural networks (advanced)

## Example: Full Workflow

```bash
# 1. Collect data
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31

# 2. Train classification for feature selection
python main.py train --use-real-data

# 3. Check feature importance
python view_model_weights.py

# 4. Train regression with top features
python train_regression.py --top-features 20

# 5. Compare different feature counts
python train_regression.py --top-features 10
python train_regression.py --top-features 30

# 6. Pick best model and use for predictions
```

## Current Results Summary

| Model Type | Output | Metric | Performance |
|------------|--------|--------|-------------|
| Logistic Regression | UP/DOWN (0/1) | Accuracy | 55.7% |
| Lasso Regression | Return % | RMSE | 2.51% |
| Lasso Regression | Return % | R² | -3.1% (needs improvement) |

The classification model works okay (better than random 50%), but the regression model needs tuning or better features!
