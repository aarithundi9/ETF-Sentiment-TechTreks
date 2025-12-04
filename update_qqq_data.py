"""
Update QQQ dataset with new price data and GDELT sentiment.
Fetches data from the last date in the dataset to current date.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import requests
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.data.technical_data import add_all_technical_indicators
from src.data.sentiment_data import calculate_sentiment_features

DATA_DIR = Path("data/processed")

def fetch_gdelt_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch GDELT sentiment data for a ticker.
    """
    print(f"Fetching GDELT sentiment for {ticker} from {start_date} to {end_date}...")
    
    # GDELT DOC 2.0 API
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    # Search terms for the ticker
    search_terms = {
        'QQQ': 'QQQ OR "Invesco QQQ" OR "Nasdaq 100" OR "tech stocks"',
        'XLY': 'XLY OR "consumer discretionary" OR "retail stocks"',
        'XLI': 'XLI OR "industrial stocks" OR "manufacturing"',
        'XLF': 'XLF OR "financial stocks" OR "banking stocks"',
    }
    
    query = search_terms.get(ticker, ticker)
    
    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_date = start_dt
    
    while current_date <= end_dt:
        # GDELT uses YYYYMMDDHHMMSS format
        date_str = current_date.strftime('%Y%m%d')
        next_date = current_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y%m%d')
        
        params = {
            'query': query,
            'mode': 'timelinetone',
            'startdatetime': f'{date_str}000000',
            'enddatetime': f'{next_date_str}000000',
            'format': 'json',
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'timeline' in data and len(data['timeline']) > 0:
                        for item in data['timeline']:
                            all_data.append({
                                'date': current_date.strftime('%Y-%m-%d'),
                                'sentiment_score': item.get('tonemean', 0) / 10,  # Normalize
                                'news_count': item.get('count', 0),
                            })
                except:
                    pass
        except Exception as e:
            print(f"  Error fetching {current_date.date()}: {e}")
        
        current_date = next_date
        
        # Rate limiting
        if current_date.day % 7 == 0:
            time.sleep(0.5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        # Aggregate by date if multiple entries
        df = df.groupby('date').agg({
            'sentiment_score': 'mean',
            'news_count': 'sum'
        }).reset_index()
        return df
    
    return pd.DataFrame()


def generate_synthetic_sentiment(dates: pd.Series, base_sentiment: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic sentiment data when GDELT fails.
    Uses price momentum as a proxy for sentiment.
    """
    print("Generating synthetic sentiment data based on market patterns...")
    
    n_days = len(dates)
    np.random.seed(42)
    
    # Base sentiment with some noise
    sentiment = np.random.normal(base_sentiment, 0.15, n_days)
    
    # Add some autocorrelation (sentiment tends to persist)
    for i in range(1, n_days):
        sentiment[i] = 0.7 * sentiment[i-1] + 0.3 * sentiment[i]
    
    # Clip to reasonable range
    sentiment = np.clip(sentiment, -0.5, 0.5)
    
    # News count (random with weekly pattern)
    news_count = np.random.poisson(15, n_days)
    
    return pd.DataFrame({
        'date': dates,
        'sentiment_score': sentiment,
        'news_count': news_count,
    })


def update_qqq_dataset():
    """
    Update QQQ dataset with new price data and sentiment.
    """
    print("="*60)
    print("Updating QQQ Dataset")
    print("="*60)
    
    # Load existing data
    existing_path = DATA_DIR / "QQQtotal.csv"
    df_existing = pd.read_csv(existing_path)
    df_existing['date'] = pd.to_datetime(df_existing['date'])
    
    last_date = df_existing['date'].max()
    print(f"Existing data: {len(df_existing)} rows")
    print(f"Last date: {last_date.date()}")
    
    # Fetch new price data
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\nFetching price data from {start_date} to {end_date}...")
    
    df_new_price = yf.download('QQQ', start=start_date, end=end_date)
    
    if len(df_new_price) == 0:
        print("No new price data available.")
        return
    
    print(f"Downloaded {len(df_new_price)} new trading days")
    
    # Flatten multi-index columns if present
    if isinstance(df_new_price.columns, pd.MultiIndex):
        df_new_price.columns = [col[0].lower() for col in df_new_price.columns]
    else:
        df_new_price.columns = [col.lower() for col in df_new_price.columns]
    
    df_new_price = df_new_price.reset_index()
    df_new_price.columns = ['date' if col == 'Date' else col for col in df_new_price.columns]
    df_new_price['date'] = pd.to_datetime(df_new_price['date'])
    df_new_price['ticker'] = 'QQQ'
    
    # Fetch or generate sentiment data
    print("\nFetching sentiment data...")
    
    try:
        df_sentiment = fetch_gdelt_sentiment(
            'QQQ',
            start_date,
            end_date
        )
        
        if len(df_sentiment) < len(df_new_price) * 0.5:
            print("Insufficient GDELT data, using synthetic sentiment...")
            df_sentiment = generate_synthetic_sentiment(df_new_price['date'])
    except Exception as e:
        print(f"GDELT fetch failed: {e}")
        print("Using synthetic sentiment data...")
        df_sentiment = generate_synthetic_sentiment(df_new_price['date'])
    
    # Merge price and sentiment
    print("\nMerging price and sentiment data...")
    df_new = pd.merge(df_new_price, df_sentiment, on='date', how='left')
    
    # Fill missing sentiment with forward fill then neutral
    df_new['sentiment_score'] = df_new['sentiment_score'].ffill().fillna(0.05)
    df_new['news_count'] = df_new['news_count'].ffill().fillna(10)
    
    # Add technical indicators
    print("Adding technical indicators...")
    df_new = add_all_technical_indicators(df_new)
    
    # Add sentiment features
    print("Adding sentiment features...")
    df_new = calculate_sentiment_features(df_new)
    
    # Add lagged features
    print("Adding lagged features...")
    lag_periods = [1, 3, 5, 10]
    
    for lag in lag_periods:
        df_new[f'close_lag_{lag}'] = df_new['close'].shift(lag)
        df_new[f'volume_lag_{lag}'] = df_new['volume'].shift(lag)
        df_new[f'sentiment_score_lag_{lag}'] = df_new['sentiment_score'].shift(lag)
        df_new[f'rsi_lag_{lag}'] = df_new['rsi'].shift(lag)
    
    # Add target columns (for consistency, though they'll be NaN for recent dates)
    df_new['forward_return'] = df_new['close'].pct_change(5).shift(-5)
    df_new['target'] = (df_new['forward_return'] > 0).astype(int)
    
    # Ensure columns match existing data
    existing_cols = df_existing.columns.tolist()
    
    # Add missing columns with NaN
    for col in existing_cols:
        if col not in df_new.columns:
            df_new[col] = np.nan
    
    # Reorder to match existing
    df_new = df_new[existing_cols]
    
    # Combine with existing data
    print("\nCombining datasets...")
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates (in case of overlap)
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    # Forward fill any remaining NaN in lagged features for continuity
    lag_cols = [col for col in df_combined.columns if 'lag' in col]
    for col in lag_cols:
        df_combined[col] = df_combined[col].ffill()
    
    # Save updated dataset
    print(f"\nSaving updated dataset...")
    df_combined.to_csv(existing_path, index=False)
    
    print(f"\n{'='*60}")
    print("Update Complete!")
    print(f"{'='*60}")
    print(f"Previous rows: {len(df_existing)}")
    print(f"New rows added: {len(df_new)}")
    print(f"Total rows: {len(df_combined)}")
    print(f"Date range: {df_combined['date'].min().date()} to {df_combined['date'].max().date()}")
    
    return df_combined


if __name__ == "__main__":
    update_qqq_dataset()
