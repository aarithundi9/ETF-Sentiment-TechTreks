# Real Data Collection Guide

This guide shows you how to collect real price and sentiment data for QQQ.

## 📊 Data Sources

### 1. **Price Data (yfinance)** ✅ Ready to use
- **Source:** Yahoo Finance
- **No API key needed**
- **Data:** OHLCV (Open, High, Low, Close, Volume)
- **Historical range:** 2010-present
- **Speed:** Very fast (~seconds)

### 2. **News Sentiment (GDELT)** ✅ Ready to use
- **Source:** GDELT Project (Global news database)
- **No API key needed** (free, rate-limited)
- **Data:** News articles + VADER sentiment scores
- **Speed:** Slow (1-2 seconds per day)
- **Note:** Can take 30+ minutes for multi-year ranges

### 3. **Reddit Sentiment** ⚠️ Requires setup
- **Source:** Reddit (wallstreetbets, stocks, investing, etc.)
- **Requires:** Reddit API credentials
- **Data:** Post titles/text + sentiment scores
- **Speed:** Medium (few minutes)

---

## 🚀 Quick Start

### Step 1: Test with Price Data Only (Fastest)

```bash
# Fetch QQQ price data from 2015-2024 (takes ~10 seconds)
python collect_real_data.py --ticker QQQ --price-only
```

This will create: `data/raw/yfinance_prices.csv`

### Step 2: Add News Sentiment (Takes time)

```bash
# Test with small date range first (recommended)
python collect_real_data.py --ticker QQQ --start 2024-11-01 --end 2024-11-30

# Then run full range (will take 30+ minutes)
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31 --no-reddit
```

This creates:
- `data/raw/gdelt_articles_QQQ.csv` (individual articles)
- `data/raw/gdelt_sentiment_QQQ.csv` (daily aggregated)

### Step 3: Add Reddit Sentiment (Optional)

**First, get Reddit API credentials:**

1. Go to: https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **Name:** ETF Sentiment Bot
   - **App type:** Select "script"
   - **Description:** Personal use for sentiment analysis
   - **About URL:** (leave blank)
   - **Redirect URI:** http://localhost:8080
4. Click "Create app"
5. Copy your credentials:
   - **Client ID:** The string under "personal use script"
   - **Client Secret:** The "secret" field

**Create `.env` file:**

```bash
# Copy example
cp .env.example .env

# Edit .env and add your Reddit credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

**Then run:**

```bash
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
```

This creates:
- `data/raw/reddit_posts_QQQ.csv` (individual posts)
- `data/raw/reddit_sentiment_QQQ.csv` (daily aggregated)

---

## 📈 After Data Collection

### Option 1: Update main.py to use real data

You'll need to modify the data loaders to read from these CSV files instead of mock data.

### Option 2: Simple test script

```python
# test_real_data.py
import pandas as pd

# Load data
prices = pd.read_csv("data/raw/yfinance_prices.csv")
gdelt = pd.read_csv("data/raw/gdelt_sentiment_QQQ.csv")

print(f"Price data: {len(prices)} days")
print(f"GDELT sentiment: {len(gdelt)} days")

# Merge on date
prices['date'] = pd.to_datetime(prices['date'])
gdelt['date'] = pd.to_datetime(gdelt['date'])
merged = prices.merge(gdelt, on=['date', 'ticker'], how='inner')

print(f"Merged data: {len(merged)} days")
print(merged.head())
```

---

## 🎯 Recommended Workflow

### Phase 1: Get Price Data (NOW)
```bash
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31 --price-only
```
**Time:** ~10 seconds  
**Result:** ~2,500 days of price data

### Phase 2: Add GDELT News (Let it run)
```bash
# Start with 1 year to test
python collect_real_data.py --ticker QQQ --start 2024-01-01 --end 2024-12-31 --no-reddit --no-price

# If that works, run full range (overnight job)
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31 --no-reddit --no-price
```
**Time:** ~5 minutes for 1 year, ~30+ minutes for 10 years  
**Result:** Daily news sentiment scores

### Phase 3: Add Reddit (Optional)
```bash
# After setting up .env with Reddit credentials
python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31 --no-price --no-gdelt
```
**Time:** ~5-10 minutes  
**Result:** Reddit sentiment scores

---

## 💡 Tips

1. **Start small:** Test with 1 month before running multi-year ranges
2. **Price data is fast:** Always fetch this first
3. **GDELT is slow but free:** Be patient, it's worth it
4. **Reddit requires setup:** But it's free once configured
5. **Save results:** All data is automatically saved to `data/raw/`

---

## 🔍 Troubleshooting

**"No data found"**
- Check date format: Must be YYYY-MM-DD
- Check ticker symbol: Use uppercase (QQQ not qqq)

**GDELT timeout**
- Normal for large date ranges
- Try smaller chunks (1 year at a time)

**Reddit error**
- Check `.env` file has correct credentials
- Verify app type is "script" not "web app"

---

## Next Steps

Once you have real data:

1. **Feature selection with Logistic Regression** (as you planned)
2. **Build Neural Network** for continuous predictions
3. **Compare:** Mock data vs Real data performance
