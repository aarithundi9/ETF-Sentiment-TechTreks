# Quick Start Guide - ETF Sentiment Analysis

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Make sure you're in the project directory
cd ETF-Sentiment-TechTreks

# Create and activate virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Generate Mock Data

```powershell
# Option A: Using the CLI
python main.py generate

# Option B: Using the module directly
python src/data/mock_data_generator.py
```

This creates:
- `data/raw/mock_prices.csv` - ~730 days of OHLCV data for QQQ, SPY, IWM
- `data/raw/mock_sentiment.csv` - Daily sentiment scores

### Step 3: Train Your First Model

```powershell
# This runs the complete pipeline:
# 1. Feature engineering
# 2. Train/test split
# 3. Model training
# 4. Evaluation
python main.py train
```

Expected output:
```
================================================================================
Training LOGISTIC_REGRESSION Model
================================================================================
Training samples: XXXX
Features: ~50+
...
✓ Training complete!
  Training accuracy: 0.XXXX
  Test accuracy: 0.XXXX
```

### Step 4: Explore the Data

```powershell
# Start Jupyter Notebook
jupyter notebook

# Open: notebooks/exploration_mock_data.ipynb
```

The notebook includes:
- Data visualization
- Technical indicator plots
- Sentiment analysis
- Model evaluation
- Feature importance

## 📊 Understanding the Output

### Generated Files

After running the pipeline, you'll have:

```
data/
├── raw/
│   ├── mock_prices.csv          # Raw OHLCV data
│   └── mock_sentiment.csv       # Raw sentiment data
├── interim/
│   ├── prices_with_technicals.csv    # Prices + technical indicators
│   └── sentiment_with_features.csv   # Sentiment + derived features
├── processed/
│   └── modeling_dataset.csv     # Final dataset for modeling
└── model_YYYYMMDD_HHMMSS.pkl   # Trained model
```

### Model Performance

You'll see metrics like:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Of predicted "up" days, how many were actually up?
- **Recall**: Of actual "up" days, how many did we predict?
- **F1-Score**: Harmonic mean of precision and recall

## 🎯 Next Steps

### 1. Experiment with Configuration

Edit `src/config/settings.py`:

```python
# Try different tickers
TICKERS = ["QQQ", "SPY", "IWM", "DIA", "TLT"]

# Adjust date range
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Modify technical indicators
TECHNICAL_INDICATORS = {
    "sma_periods": [10, 20, 50, 200],  # Add 200-day SMA
    # ...
}
```

### 2. Add Your Own Data

Place CSV files in `data/raw/`:

**user_prices.csv**:
```csv
date,ticker,open,high,low,close,volume
2023-01-01,QQQ,300.0,305.0,299.0,304.0,50000000
```

**user_sentiment.csv**:
```csv
date,ticker,sentiment_score,news_count
2023-01-01,QQQ,0.25,15
```

Then update the data fetcher:
```python
from src.data.technical_data import TechnicalDataFetcher

fetcher = TechnicalDataFetcher(source="csv")
prices = fetcher.fetch_ohlcv()
```

### 3. Use Real Data (yfinance)

Uncomment the yfinance code in `src/data/technical_data.py`:

```python
def _fetch_yfinance_data(self, tickers, start_date, end_date):
    import yfinance as yf
    # ... (uncomment the implementation)
```

Then use:
```python
fetcher = TechnicalDataFetcher(source="yfinance")
prices = fetcher.fetch_ohlcv()
```

### 4. Integrate Real Sentiment Sources

Add your API keys to `.env`:
```bash
NEWS_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_id
```

Uncomment the API integration code in `src/data/sentiment_data.py`

### 5. Try Different Models

Modify `src/models/train_model.py` to add Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier

def _create_model(self):
    if self.model_type == "logistic_regression":
        return LogisticRegression(...)
    elif self.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
```

## 🧪 Running Tests

```powershell
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🔍 Troubleshooting

### Import Errors

If you see import errors, make sure:
1. You're in the project root directory
2. Virtual environment is activated
3. All dependencies are installed: `pip install -r requirements.txt`

### Module Not Found

Some modules (like dotenv) might not be installed. Install missing packages:
```powershell
pip install python-dotenv
```

### Data Not Found

If you see "Mock data not found", run:
```powershell
python main.py generate
```

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Code Examples**: Check `notebooks/exploration_mock_data.ipynb`
- **Configuration**: Review `src/config/settings.py`
- **Tests**: Look at `tests/` for usage examples

## 💡 Tips

1. **Start Simple**: Use mock data first to understand the pipeline
2. **Check Config**: Always review `src/config/settings.py` before running
3. **Save Models**: Trained models are automatically saved with timestamps
4. **Version Control**: Use git to track changes to configuration and code
5. **Test Often**: Run tests after making changes: `pytest tests/`

## 🤝 Get Help

- Open an issue on GitHub
- Check the documentation in code docstrings
- Review the test files for usage examples

---

**You're all set! Happy modeling! 🎉**
