# ETF Sentiment Analysis - Project Overview

## 📋 Complete File Structure

```
ETF-Sentiment-TechTreks/
│
├── main.py                      # CLI entry point
├── README.md                    # Complete documentation
├── QUICKSTART.md               # Quick start guide
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
│
├── data/                       # Data directory
│   ├── raw/                   # Raw data
│   │   ├── .gitkeep
│   │   ├── mock_prices.csv    # Generated mock OHLCV data
│   │   └── mock_sentiment.csv # Generated mock sentiment
│   ├── interim/               # Intermediate processed data
│   │   └── .gitkeep
│   └── processed/             # Final modeling datasets
│       └── .gitkeep
│
├── notebooks/                  # Jupyter notebooks
│   └── exploration_mock_data.ipynb  # Complete data exploration
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── config/                # Configuration
│   │   ├── __init__.py
│   │   └── settings.py        # Global settings & paths
│   │
│   ├── data/                  # Data acquisition & processing
│   │   ├── __init__.py
│   │   ├── mock_data_generator.py    # Mock data generation
│   │   ├── technical_data.py         # Technical indicators
│   │   ├── sentiment_data.py         # Sentiment analysis
│   │   └── user_data_loader.py       # User CSV loader
│   │
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py  # Feature pipeline
│   │
│   └── models/                # Machine learning models
│       ├── __init__.py
│       ├── train_model.py     # Model training
│       └── evaluate_model.py  # Model evaluation
│
└── tests/                      # Unit tests
    ├── test_data_pipeline.py       # Data pipeline tests
    └── test_feature_pipeline.py    # Feature engineering tests
```

## 🎯 Key Components

### 1. Configuration (`src/config/settings.py`)
- Global settings for tickers, dates, paths
- Technical indicator parameters
- Model hyperparameters
- Single source of truth

### 2. Data Pipeline (`src/data/`)
- **mock_data_generator.py**: Generate realistic mock data
- **technical_data.py**: Fetch OHLCV data, calculate indicators
- **sentiment_data.py**: Fetch/analyze sentiment data
- **user_data_loader.py**: Load user-provided CSV files

### 3. Features (`src/features/`)
- **build_features.py**: Complete feature engineering pipeline
  - Merge price & sentiment
  - Create lagged features
  - Generate target variable
  - Train/test split

### 4. Models (`src/models/`)
- **train_model.py**: Model training with cross-validation
- **evaluate_model.py**: Comprehensive evaluation metrics

### 5. Entry Points

#### CLI (`main.py`)
```powershell
python main.py generate   # Generate mock data
python main.py pipeline   # Run feature engineering
python main.py train      # Train model (full pipeline)
python main.py config     # Show configuration
```

#### Individual Modules
```powershell
python src/data/mock_data_generator.py
python src/features/build_features.py
python src/models/train_model.py
```

#### Jupyter Notebook
```powershell
jupyter notebook
# Open: notebooks/exploration_mock_data.ipynb
```

## 🔄 Workflow

### Standard Workflow
```
1. Generate Mock Data
   ↓
2. Feature Engineering
   ↓
3. Train Model
   ↓
4. Evaluate
```

### Development Workflow
```
1. Edit Configuration (src/config/settings.py)
   ↓
2. Test with Mock Data
   ↓
3. Integrate Real Data Sources
   ↓
4. Iterate on Features
   ↓
5. Experiment with Models
```

## 📊 Data Flow

```
Raw Data Sources:
├── Mock Data (development)
├── yfinance (real prices)
├── News APIs (sentiment)
└── User CSV files (custom)
         ↓
    Feature Engineering:
    ├── Technical indicators (SMA, EMA, RSI, MACD, BB)
    ├── Sentiment features (scores, moving averages)
    ├── Lagged features (1, 3, 5, 10 periods)
    └── Target variable (binary: up/down)
         ↓
    Machine Learning:
    ├── Train/test split (80/20, time-based)
    ├── Scaling (StandardScaler)
    ├── Model training (Logistic Regression baseline)
    └── Cross-validation (5-fold)
         ↓
    Evaluation:
    ├── Accuracy, Precision, Recall, F1
    ├── Confusion Matrix
    ├── Per-ticker metrics
    └── Trading performance simulation
```

## 🎛️ Configuration Options

### Tickers
```python
TICKERS = ["QQQ", "SPY", "IWM"]  # Easy to extend
```

### Technical Indicators
```python
TECHNICAL_INDICATORS = {
    "sma_periods": [5, 10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bollinger_period": 20, "bollinger_std": 2,
}
```

### Model Settings
```python
MODEL_CONFIG = {
    "model_type": "logistic_regression",
    "random_state": 42,
    "hyperparameters": {"C": 1.0, "max_iter": 1000},
}
```

## 🧩 Extension Points

### Add New Data Source
1. Extend `TechnicalDataFetcher` or `SentimentDataFetcher`
2. Add new source in `_fetch_X_data()` method
3. Update configuration

### Add New Technical Indicator
1. Create function in `technical_data.py`
2. Add to `add_all_technical_indicators()`
3. Update `TECHNICAL_INDICATORS` config

### Add New Model
1. Extend `ETFPricePredictor._create_model()`
2. Add model type to config
3. Optional: Custom evaluation metrics

### Add New Feature
1. Create feature function in `build_features.py`
2. Add to `create_feature_pipeline()`

## 📈 Performance Metrics

### Model Metrics
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision/recall

### Trading Metrics
- **Strategy Return**: Model-based trading return
- **Buy & Hold Return**: Benchmark return
- **Outperformance**: Strategy vs. benchmark
- **Number of Trades**: Trading frequency

## 🔐 Environment Variables

Create `.env` from `.env.example`:
```bash
NEWS_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
TWITTER_BEARER_TOKEN=your_token
```

## 🧪 Testing

```powershell
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_data_pipeline.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Best Practices

1. **Always use mock data first** - Test pipeline before real data
2. **Check configuration** - Review `settings.py` before running
3. **Version control** - Git commit after major changes
4. **Test changes** - Run pytest after modifications
5. **Document** - Update docstrings for new features

## 🚀 Quick Commands

```powershell
# Setup
pip install -r requirements.txt

# Generate data
python main.py generate

# Full pipeline + training
python main.py train

# Run tests
pytest tests/ -v

# Jupyter
jupyter notebook notebooks/exploration_mock_data.ipynb

# View config
python main.py config
```

## 📚 Documentation Locations

- **README.md**: Complete project documentation
- **QUICKSTART.md**: 5-minute setup guide
- **This file**: Project structure overview
- **Docstrings**: In-code documentation
- **Notebook**: Interactive exploration

## 🎓 Learning Path

1. ✅ Read QUICKSTART.md
2. ✅ Generate mock data
3. ✅ Run complete pipeline
4. ✅ Explore Jupyter notebook
5. ✅ Review configuration options
6. ✅ Read module docstrings
7. ✅ Run tests to see examples
8. ✅ Modify configuration
9. ✅ Add custom features
10. ✅ Integrate real data

---

**You have everything you need to start building! 🎉**
