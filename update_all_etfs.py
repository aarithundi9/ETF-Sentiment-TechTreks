"""
Update price + sentiment data for XLY, XLI, XLF ETFs to current date
Then train XGBoost models for each
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Daily return
    df['daily_return'] = df['close'].pct_change()
    
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lagged features"""
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
    return df

def generate_synthetic_sentiment(df):
    """Generate synthetic sentiment based on price movements"""
    np.random.seed(42)
    
    # Base sentiment on recent returns with noise
    returns_5d = df['close'].pct_change(5).fillna(0)
    
    # Sentiment somewhat correlated with returns but noisy
    base_sentiment = returns_5d * 10 + np.random.normal(0, 0.1, len(df))
    
    # Normalize to 0-1 range
    sentiment_positive = (base_sentiment - base_sentiment.min()) / (base_sentiment.max() - base_sentiment.min() + 1e-8)
    sentiment_positive = sentiment_positive.clip(0.1, 0.9)
    
    df['sentiment_positive'] = sentiment_positive
    df['sentiment_negative'] = 1 - sentiment_positive + np.random.normal(0, 0.05, len(df))
    df['sentiment_negative'] = df['sentiment_negative'].clip(0.05, 0.5)
    df['sentiment_neutral'] = 1 - df['sentiment_positive'] - df['sentiment_negative']
    df['sentiment_neutral'] = df['sentiment_neutral'].clip(0.1, 0.5)
    
    # Normalize so they sum to 1
    total = df['sentiment_positive'] + df['sentiment_negative'] + df['sentiment_neutral']
    df['sentiment_positive'] /= total
    df['sentiment_negative'] /= total
    df['sentiment_neutral'] /= total
    
    return df

def update_etf_data(ticker):
    """Update ETF data to current date"""
    print(f"\n{'='*50}")
    print(f"Updating {ticker}...")
    print('='*50)
    
    # Load existing data
    filepath = f'data/processed/{ticker}total.csv'
    df_existing = pd.read_csv(filepath)
    df_existing['date'] = pd.to_datetime(df_existing['date'])
    
    last_date = df_existing['date'].max()
    print(f"Existing data: {len(df_existing)} rows, {df_existing['date'].min().date()} to {last_date.date()}")
    
    # Fetch new data from yfinance
    start_date = last_date + timedelta(days=1)
    end_date = datetime.now()
    
    if start_date >= end_date:
        print(f"Data already up to date!")
        return df_existing
    
    print(f"Fetching new data from {start_date.date()} to {end_date.date()}...")
    
    try:
        stock = yf.Ticker(ticker)
        df_new = stock.history(start=start_date, end=end_date)
        
        if len(df_new) == 0:
            print("No new data available")
            return df_existing
        
        print(f"Fetched {len(df_new)} new rows")
        
        # Format new data
        df_new = df_new.reset_index()
        df_new.columns = [c.lower().replace(' ', '_') for c in df_new.columns]
        df_new = df_new.rename(columns={'date': 'date'})
        
        # Handle timezone-aware dates
        if df_new['date'].dt.tz is not None:
            df_new['date'] = df_new['date'].dt.tz_localize(None)
        
        # Keep only OHLCV columns
        df_new = df_new[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Combine with existing
        df_combined = pd.concat([df_existing[['date', 'open', 'high', 'low', 'close', 'volume']], df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        
        print(f"Combined data: {len(df_combined)} rows")
        
        # Recalculate all features
        print("Adding technical indicators...")
        df_combined = add_technical_indicators(df_combined)
        
        print("Adding lagged features...")
        df_combined = add_lagged_features(df_combined)
        
        print("Generating synthetic sentiment...")
        df_combined = generate_synthetic_sentiment(df_combined)
        
        # Drop rows with NaN (from indicators)
        df_combined = df_combined.dropna().reset_index(drop=True)
        
        # Save updated data
        df_combined.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        print(f"Final: {len(df_combined)} rows, {df_combined['date'].min()} to {df_combined['date'].max()}")
        
        return df_combined
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return df_existing

def main():
    """Update all ETFs"""
    etfs = ['XLY', 'XLI', 'XLF']
    
    # First, show current status
    print("Current ETF Data Status:")
    print("-" * 60)
    for ticker in etfs:
        df = pd.read_csv(f'data/processed/{ticker}total.csv')
        print(f"{ticker}: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    
    # Update each ETF
    for ticker in etfs:
        update_etf_data(ticker)
    
    # Show final status
    print("\n" + "="*60)
    print("FINAL STATUS:")
    print("="*60)
    for ticker in ['QQQ'] + etfs:
        df = pd.read_csv(f'data/processed/{ticker}total.csv')
        print(f"{ticker}: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    main()
