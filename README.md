# ETF Sentiment Analysis & Price Prediction

A complete Python project for predicting ETF price movements using historical OHLCV data, technical indicators, and sentiment analysis from news and social media.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project provides a **production-ready framework** for ETF price movement prediction that combines:

- **Historical Price Data**: OHLCV (Open, High, Low, Close, Volume) with technical indicators
- **Sentiment Analysis**: News and social media sentiment scores
- **Machine Learning**: Baseline models (Logistic Regression) with easy extensibility
- **Mock Data**: Built-in mock data generator for development and testing

**Current Focus**: Clean architecture and robust ETL pipeline (not model performance)

### Supported ETFs

- **QQQ** - Invesco QQQ Trust (Nasdaq-100)
- **SPY** - SPDR S&P 500 ETF Trust
- **IWM** - iShares Russell 2000 ETF

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/aarithundi9/ETF-Sentiment-TechTreks.git
cd ETF-Sentiment-TechTreks

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Mock Data

```powershell
# Generate mock price and sentiment data
python src/data/mock_data_generator.py
```

This creates:
- `data/raw/mock_prices.csv` - Historical OHLCV data (~2 years)
- `data/raw/mock_sentiment.csv` - Daily sentiment scores

### 3. Run the Complete Pipeline

```powershell
# Run feature engineering pipeline
python src/features/build_features.py

# Train a baseline model
python src/models/train_model.py
```

### 4. Explore in Jupyter

```powershell
# Start Jupyter
jupyter notebook

# Open the exploration notebook
# notebooks/exploration_mock_data.ipynb
```

## 📁 Project Structure

```
ETF-Sentiment-TechTreks/
├── data/
│   ├── raw/              # Raw data (CSV files, mock data)
│   ├── interim/          # Intermediate processed data
│   └── processed/        # Final modeling datasets
├── notebooks/
│   └── exploration_mock_data.ipynb  # Data exploration notebook
├── src/
│   ├── config/
│   │   └── settings.py   # Global configuration
│   ├── data/
│   │   ├── mock_data_generator.py    # Mock data generation
│   │   ├── technical_data.py         # Technical indicators
│   │   ├── sentiment_data.py         # Sentiment analysis
│   │   └── user_data_loader.py       # Load user CSV files
│   ├── features/
│   │   └── build_features.py         # Feature engineering
│   └── models/
│       ├── train_model.py            # Model training
│       └── evaluate_model.py         # Model evaluation
├── tests/
│   ├── test_data_pipeline.py         # Data pipeline tests
│   └── test_feature_pipeline.py      # Feature engineering tests
├── .env.example          # Environment variables template
├── .gitignore
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Configuration

All configuration is centralized in `src/config/settings.py`:

```python
# Tickers to analyze
TICKERS = ["QQQ", "SPY", "IWM"]

# Date range
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

# Technical indicators
TECHNICAL_INDICATORS = {
    "sma_periods": [5, 10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    # ... more settings
}
```

### Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
# News API
NEWS_API_KEY=your_api_key_here

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret

# Twitter API
TWITTER_BEARER_TOKEN=your_token
```

## 📊 Features

### Technical Indicators

- **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)
- **RSI** (Relative Strength Index): 14-period
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**: 20-period, 2 std dev

### Sentiment Features

- **Sentiment Score**: -1 (very negative) to +1 (very positive)
- **News Count**: Daily article/post count
- **Sentiment Moving Averages**: 3, 7, 14-day
- **Sentiment Momentum**: Change in sentiment
- **Sentiment Volatility**: 7-day rolling standard deviation

### Target Variable

- **Binary Classification**: 1 if price goes up next day, 0 otherwise
- **Customizable**: Easy to change to continuous (regression) or multi-day forecasts

## 🤖 Models

### Baseline Model (Current)

- **Algorithm**: Logistic Regression
- **Features**: ~50+ technical + sentiment features
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Extending to Other Models

The framework is designed for easy model swapping:

```python
# In src/models/train_model.py
class ETFPricePredictor:
    def _create_model(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(...)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(...)
        elif self.model_type == "xgboost":
            return XGBClassifier(...)
```

## 📈 Usage Examples

### Generate Mock Data

```python
from src.data.mock_data_generator import generate_all_mock_data

prices_df, sentiment_df = generate_all_mock_data(save=True, verbose=True)
print(f"Generated {len(prices_df)} price records")
```

### Load Real Data (When Ready)

```python
from src.data.technical_data import TechnicalDataFetcher

# Uncomment yfinance code in technical_data.py first
fetcher = TechnicalDataFetcher(source="yfinance")
prices_df = fetcher.fetch_ohlcv(
    tickers=["QQQ", "SPY"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Add User CSV Data

Place your CSV files in `data/raw/`:

**Price Data** (`user_prices.csv`):
```csv
date,ticker,open,high,low,close,volume
2023-01-01,QQQ,300.0,305.0,299.0,304.0,50000000
```

**Sentiment Data** (`user_sentiment.csv`):
```csv
date,ticker,sentiment_score,news_count
2023-01-01,QQQ,0.25,15
```

### Train a Custom Model

```python
from src.features.build_features import create_feature_pipeline, prepare_train_test_split
from src.models.train_model import ETFPricePredictor

# Prepare data
df = create_feature_pipeline(use_mock_data=True)
X_train, X_test, y_train, y_test = prepare_train_test_split(df)

# Train model
model = ETFPricePredictor(model_type='logistic_regression')
model.fit(X_train, y_train)

# Evaluate
from src.models.evaluate_model import evaluate_model
metrics = evaluate_model(model, X_test, y_test)

# Save model
model.save('data/my_model.pkl')
```

## 🧪 Testing

Run the test suite:

```powershell
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔮 Future Enhancements

### Data Sources
- [ ] Integrate real-time yfinance data
- [ ] News API integration (NewsAPI, Alpha Vantage)
- [ ] Reddit sentiment (via PRAW)
- [ ] Twitter/X sentiment
- [ ] Alternative data sources (Quiver Quantitative, etc.)

### Features
- [ ] More technical indicators (ADX, ATR, etc.)
- [ ] Market regime detection
- [ ] Volatility features (GARCH, realized volatility)
- [ ] Cross-asset features (VIX, bonds, crypto correlation)

### Models
- [ ] Random Forest
- [ ] XGBoost / LightGBM
- [ ] Neural Networks (LSTM, Transformer)
- [ ] Ensemble methods
- [ ] Online learning for real-time updates

### Infrastructure
- [ ] Real-time prediction API (FastAPI)
- [ ] Automated daily data updates
- [ ] Model monitoring and retraining
- [ ] Backtesting with realistic costs
- [ ] Dashboard (Streamlit/Dash)

## 📚 Documentation

- **Configuration**: See `src/config/settings.py` for all settings
- **Data Pipeline**: See docstrings in `src/data/`
- **Feature Engineering**: See `src/features/build_features.py`
- **Model API**: See `src/models/train_model.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is NOT financial advice. 

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Always do your own research before investing
- Consult with a qualified financial advisor

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Aari Thundi** - [aarithundi9](https://github.com/aarithundi9)

## 🙏 Acknowledgments

- **Technical Indicators**: Inspired by TA-Lib and pandas-ta
- **Sentiment Analysis**: Using VADER Sentiment
- **Data Sources**: yfinance, NewsAPI (when integrated)
- **ML Framework**: scikit-learn

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Trading! 📈**
