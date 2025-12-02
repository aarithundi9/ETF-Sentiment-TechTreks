# Multi-Ticker Analysis Guide

This guide shows how to extend the pipeline to analyze multiple ETFs simultaneously.

## Quick Example: Add SPY

### 1. Collect Data for SPY

```powershell
# Collect SPY data (same command as QQQ)
python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
```

Expected output:
```
Collecting data for SPY from 2015-01-01 to 2024-12-31
[1/3] Fetching price data from yfinance...
✓ Downloaded 2515 days of price data
✓ Saved to data/raw/yfinance_prices.csv

[2/3] Fetching news sentiment from GDELT...
Processing: 100%|████████████████████| 2922/2922
✓ Collected 2822 articles
✓ Saved to data/raw/gdelt_sentiment_SPY.csv
```

### 2. Update Configuration

Edit `src/config/settings.py`:

```python
# Change from:
TICKERS = ["QQQ"]

# To:
TICKERS = ["SPY"]
# or both:
TICKERS = ["QQQ", "SPY"]
```

### 3. Train Model

```powershell
python main.py train --use-real-data
```

That's it! The pipeline automatically handles the new ticker.

## Analyzing Multiple Tickers Simultaneously

### Option 1: Combined Model (Single Ticker at a Time)

Train separate models for each ticker:

```powershell
# QQQ model
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
# Edit settings.py: TICKERS = ["QQQ"]
python main.py train --use-real-data

# SPY model
python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
# Edit settings.py: TICKERS = ["SPY"]
python main.py train --use-real-data

# IWM model
python collect_real_data.py --ticker IWM --start 2015-01-01 --end 2024-12-31
# Edit settings.py: TICKERS = ["IWM"]
python main.py train --use-real-data
```

### Option 2: Unified Model (All Tickers Together)

Train one model on all tickers:

```powershell
# Collect data for all tickers
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
python collect_real_data.py --ticker IWM --start 2015-01-01 --end 2024-12-31

# Edit settings.py
# TICKERS = ["QQQ", "SPY", "IWM"]

# Train unified model
python main.py train --use-real-data
```

**Advantages:**
- More training data (3x samples)
- Learns cross-ticker patterns
- Single model to maintain

**Considerations:**
- May need ticker-specific features (relative performance, correlation)
- Different volatility profiles may confuse model
- Test performance on each ticker separately

## Batch Data Collection Script

Create `collect_all_tickers.py`:

```python
import subprocess
import sys

TICKERS = ["QQQ", "SPY", "IWM", "DIA", "VTI"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

for ticker in TICKERS:
    print(f"\n{'='*80}")
    print(f"Collecting data for {ticker}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "collect_real_data.py",
        "--ticker", ticker,
        "--start", START_DATE,
        "--end", END_DATE,
        "--no-reddit",  # Skip Reddit if no credentials
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Failed to collect data for {ticker}")
    else:
        print(f"✓ Successfully collected data for {ticker}")

print("\n" + "="*80)
print("Data collection complete!")
print("="*80)
```

Run it:
```powershell
python collect_all_tickers.py
```

## Comparing Performance Across Tickers

Create `compare_tickers.py`:

```python
"""Compare model performance across different tickers."""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.features.build_features import create_feature_pipeline, prepare_train_test_split
from src.models.train_model import train_baseline_model
from src.config import settings

TICKERS_TO_TEST = ["QQQ", "SPY", "IWM"]

results = []

for ticker in TICKERS_TO_TEST:
    print(f"\n{'='*80}")
    print(f"Testing {ticker}")
    print(f"{'='*80}")
    
    # Update settings
    settings.TICKERS = [ticker]
    
    try:
        # Run pipeline
        df = create_feature_pipeline(use_mock_data=False, save_interim=False)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df)
        
        # Train model
        model, metrics = train_baseline_model(
            X_train, y_train, X_test, y_test,
            perform_cv=False,
            save_model=False,
        )
        
        results.append({
            'ticker': ticker,
            'samples': len(df),
            'features': X_train.shape[1],
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        })
        
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Display results
print("\n" + "="*80)
print("TICKER COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('ticker_comparison_results.csv', index=False)
print(f"\n✓ Results saved to ticker_comparison_results.csv")
```

## Advanced: Cross-Ticker Features

Add features that compare tickers to each other:

```python
def add_cross_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features comparing tickers."""
    
    # Example: Relative performance
    if "QQQ" in df['ticker'].values and "SPY" in df['ticker'].values:
        # Get returns for each ticker
        qqq_returns = df[df['ticker'] == 'QQQ'][['date', 'close']].rename(
            columns={'close': 'qqq_close'}
        )
        spy_returns = df[df['ticker'] == 'SPY'][['date', 'close']].rename(
            columns={'close': 'spy_close'}
        )
        
        # Merge
        merged = df.merge(qqq_returns, on='date', how='left')
        merged = merged.merge(spy_returns, on='date', how='left')
        
        # Create relative performance feature
        merged['qqq_spy_ratio'] = merged['qqq_close'] / merged['spy_close']
        
        return merged
    
    return df
```

## Recommended Multi-Ticker Workflow

### Step 1: Collect All Data (One Time)

```powershell
# Major ETFs
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
python collect_real_data.py --ticker SPY --start 2015-01-01 --end 2024-12-31
python collect_real_data.py --ticker IWM --start 2015-01-01 --end 2024-12-31
python collect_real_data.py --ticker DIA --start 2015-01-01 --end 2024-12-31

# Sector ETFs (optional)
python collect_real_data.py --ticker XLK --start 2015-01-01 --end 2024-12-31  # Tech
python collect_real_data.py --ticker XLF --start 2015-01-01 --end 2024-12-31  # Finance
```

### Step 2: Train Individual Models

```python
# train_all_tickers.py
import subprocess
import sys

TICKERS = ["QQQ", "SPY", "IWM", "DIA"]

for ticker in TICKERS:
    print(f"\nTraining model for {ticker}...")
    
    # Update settings.py programmatically
    with open('src/config/settings.py', 'r') as f:
        content = f.read()
    
    # Replace TICKERS line
    content = content.replace(
        'TICKERS = [', 
        f'TICKERS = ["{ticker}"]  # Auto-set by train_all_tickers.py\n# Original: ['
    )
    
    with open('src/config/settings.py', 'w') as f:
        f.write(content)
    
    # Train
    result = subprocess.run([
        sys.executable, "main.py", "train", "--use-real-data"
    ])
    
    if result.returncode == 0:
        print(f"✓ {ticker} training complete")
    else:
        print(f"❌ {ticker} training failed")
```

### Step 3: Compare Results

```powershell
python compare_tickers.py
```

### Step 4: Choose Best Approach

Based on results, decide:
- **Separate models**: If tickers behave very differently
- **Unified model**: If patterns are similar across tickers
- **Ensemble**: Combine predictions from multiple models

## Data Files Organization

After collecting multiple tickers:

```
data/raw/
├── yfinance_prices.csv          # Contains ALL tickers
├── gdelt_sentiment_QQQ.csv      # QQQ sentiment only
├── gdelt_sentiment_SPY.csv      # SPY sentiment only
├── gdelt_sentiment_IWM.csv      # IWM sentiment only
├── gdelt_sentiment_DIA.csv      # DIA sentiment only
├── gdelt_articles_QQQ.csv       # Optional: detailed articles
├── gdelt_articles_SPY.csv
└── ...
```

The `yfinance_prices.csv` file contains data for ALL tickers in a single file:
```csv
date,ticker,open,high,low,close,volume
2015-01-02,QQQ,105.12,106.34,104.89,106.01,45123456
2015-01-02,SPY,203.45,204.56,203.12,204.23,67890123
...
```

GDELT sentiment files are separate per ticker (due to API limitations).

## Troubleshooting Multi-Ticker Issues

### Issue: "No sentiment data found for ticker X"

**Solution:**
1. Verify you ran `collect_real_data.py` for that ticker
2. Check `data/raw/` for `gdelt_sentiment_X.csv`
3. Re-run collection if missing

### Issue: Different date ranges for different tickers

**Solution:**
The pipeline automatically aligns dates when merging. It will use the intersection of available dates across all tickers.

### Issue: Poor performance on some tickers

**Solution:**
- Check if ticker has enough trading history
- Verify sentiment data quality (some tickers have less news coverage)
- Consider training separate models for each ticker
- Adjust prediction horizon (forward_period)

## Supported Tickers

The pipeline works with any ticker available in:
1. **yfinance** (Yahoo Finance) - most US stocks and ETFs
2. **GDELT** - news coverage required

### Popular ETFs for Testing:

**Market Indices:**
- QQQ - Nasdaq-100
- SPY - S&P 500
- IWM - Russell 2000
- DIA - Dow Jones

**Sector ETFs:**
- XLK - Technology
- XLF - Financial
- XLE - Energy
- XLV - Healthcare
- XLI - Industrial

**International:**
- EFA - EAFE (Europe, Asia)
- EEM - Emerging Markets
- VEU - All-World ex-US

## Next Steps

1. **Collect data for 3-5 tickers** you're interested in
2. **Train individual models** for each
3. **Compare performance** across tickers
4. **Experiment with unified model** using all tickers
5. **Add cross-ticker features** for better predictions
6. **Deploy best-performing models** to production

For questions or issues, check the main USAGE_GUIDE.md or open an issue on GitHub.
