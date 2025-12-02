# ETF Sentiment Analysis - Usage Guide

This guide shows how to use the pipeline with both **mock data** (for testing) and **real data** (for production).

## Quick Start

### Option 1: Mock Data (for testing/development)

```bash
# Generate mock data
python main.py generate

# Run feature engineering pipeline
python main.py pipeline

# Train and evaluate model
python main.py train
```

### Option 2: Real Data (for production)

```bash
# Step 1: Collect real data (required only once, or to update)
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31

# Step 2: Run pipeline with real data
python main.py pipeline --use-real-data

# Step 3: Train model with real data
python main.py train --use-real-data
```

## Detailed Workflow

### 1. Data Collection (Real Data Only)

The `collect_real_data.py` script fetches data from:
- **yfinance**: Historical OHLCV price data (free, no API key required)
- **GDELT**: News articles and sentiment (free, no API key required)
- **Reddit**: Social media sentiment (requires API credentials - optional)

**Basic usage:**
```bash
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
```

**Collect for multiple tickers:**
```bash
# QQQ
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31

# SPY
python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31

# IWM
python collect_real_data.py --ticker IWM --start 2015-01-01 --end 2024-12-31
```

**Advanced options:**
```bash
# Collect only price data (skip sentiment)
python collect_real_data.py --ticker QQQ --price-only

# Collect only sentiment data (skip price)
python collect_real_data.py --ticker QQQ --no-price

# Skip Reddit (if you don't have API credentials)
python collect_real_data.py --ticker QQQ --no-reddit

# Update data (collect recent data only)
python collect_real_data.py --ticker QQQ --start 2024-11-01 --end 2024-12-31
```

**Expected outputs:**
- `data/raw/yfinance_prices.csv` - Historical OHLCV data
- `data/raw/gdelt_sentiment_{ticker}.csv` - Daily aggregated sentiment
- `data/raw/gdelt_articles_{ticker}.csv` - Individual article data (optional)

**Note on date ranges:**
- yfinance: Data available from ~2015 onwards for most ETFs
- GDELT: News sentiment available from ~2017 onwards
- The pipeline will automatically align dates when merging

### 2. Feature Engineering Pipeline

The pipeline performs:
1. Load price/sentiment data
2. Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. Calculate sentiment features (rolling averages, momentum)
4. Merge datasets on date/ticker
5. Create lagged features (1, 3, 5, 10 days)
6. Create target variable (5-day forward return - weekly prediction)
7. Save processed dataset

**Mock data (for testing):**
```bash
python main.py pipeline
```

**Real data (for production):**
```bash
python main.py pipeline --use-real-data
```

**Output:**
- `data/processed/modeling_dataset.csv` - Ready for ML training
- `data/interim/prices_with_technicals.csv` - Intermediate file
- `data/interim/sentiment_with_features.csv` - Intermediate file

### 3. Model Training

The training pipeline:
1. Runs feature engineering (or loads existing dataset)
2. Splits into train/test (80/20, time-based)
3. Trains logistic regression with cross-validation
4. Evaluates on test set
5. Saves trained model

**Mock data:**
```bash
python main.py train
```

**Real data:**
```bash
python main.py train --use-real-data
```

**Output:**
- `data/model_YYYYMMDD_HHMMSS.pkl` - Trained model with scaler
- Performance metrics printed to console

### 4. View Configuration

```bash
python main.py config
```

## Configuration

Edit `src/config/settings.py` to customize:

```python
# Which tickers to analyze
TICKERS = ["QQQ"]  # Add more: ["QQQ", "SPY", "IWM"]

# Date range for analysis
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Prediction horizon
FEATURE_CONFIG = {
    'forward_period': 5,  # 5 = weekly, 1 = daily
    'target_type': 'binary',  # 'binary' or 'continuous'
    'lags': [1, 3, 5, 10],  # Lagged features
}

# Technical indicators
TECHNICAL_CONFIG = {
    'sma_periods': [5, 10, 20, 50],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'macd_periods': (12, 26, 9),
    'bollinger_period': 20,
    'bollinger_std': 2,
}
```

## Reproducibility for Different Tickers

To analyze a new ticker (e.g., SPY):

1. **Collect data:**
   ```bash
   python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
   ```

2. **Update config:**
   Edit `src/config/settings.py`:
   ```python
   TICKERS = ["SPY"]  # or ["QQQ", "SPY"] for both
   ```

3. **Run pipeline:**
   ```bash
   python main.py train --use-real-data
   ```

That's it! The same pipeline works for any ticker.

## File Structure

```
data/
├── raw/                              # Raw data from collection
│   ├── yfinance_prices.csv          # OHLCV data for all tickers
│   ├── gdelt_sentiment_QQQ.csv      # Daily sentiment for QQQ
│   ├── gdelt_sentiment_SPY.csv      # Daily sentiment for SPY
│   └── ...
├── interim/                          # Intermediate processed data
│   ├── prices_with_technicals.csv
│   └── sentiment_with_features.csv
├── processed/                        # Final datasets ready for ML
│   └── modeling_dataset.csv
└── model_*.pkl                       # Trained models
```

## Troubleshooting

### "GDELT sentiment not found"
- Run `python collect_real_data.py --ticker QQQ --no-price` to collect sentiment
- GDELT data is only available from 2017 onwards

### "No matching records after merge"
- Check date ranges align (sentiment starts 2017, price can start earlier)
- Verify ticker symbols match exactly (case-sensitive)

### "SSL certificate error" or "Package installation failed"
- Make sure you're using Windows Python (not MSYS2/Cygwin)
- Activate virtual environment: `.\venv\Scripts\Activate.ps1`

### Poor model performance
- Try mock data first to verify pipeline works: `python main.py train`
- Real data may have lower accuracy (markets are hard to predict!)
- Experiment with different features, lags, or prediction horizons

## Performance Comparison

Typical results on QQQ (2015-2024):

| Dataset | Test Accuracy | F1-Score | Notes |
|---------|---------------|----------|-------|
| Mock Data | ~0.65 | ~0.70 | Synthetic, correlated by design |
| Real Data | ~0.55 | ~0.60 | Actual market data, more challenging |

**Note:** The mock data has artificially strong correlations between sentiment and price, so performance is higher. Real market data is much noisier and harder to predict.

## Next Steps

1. **Feature Selection:**
   - Examine feature importance from trained model
   - Remove low-importance features
   - Re-train and compare performance

2. **Advanced Models:**
   - Try different models (Random Forest, XGBoost, Neural Networks)
   - Tune hyperparameters
   - Ensemble methods

3. **More Data Sources:**
   - Add Reddit sentiment (requires API setup)
   - Add Twitter/X sentiment
   - Add macroeconomic indicators

4. **Continuous Predictions:**
   - Change `target_type` to 'continuous' for regression
   - Predict actual return percentage instead of direction

5. **Production Deployment:**
   - Schedule data collection (daily cron job)
   - Retrain model periodically
   - Generate predictions for trading
