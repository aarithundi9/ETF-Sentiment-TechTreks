"""Split yfinance_prices.csv into separate files per ticker."""

import pandas as pd
from pathlib import Path

# Read the combined file
df = pd.read_csv('data/raw/yfinance_prices.csv')

print("=" * 80)
print("SPLITTING PRICE DATA BY TICKER")
print("=" * 80)

# Get unique tickers
tickers = df['ticker'].unique()
print(f"Found {len(tickers)} tickers: {', '.join(tickers)}\n")

# Split and save each ticker
for ticker in tickers:
    ticker_df = df[df['ticker'] == ticker].copy()
    
    output_path = Path(f"data/raw/yfinance_prices_{ticker}.csv")
    ticker_df.to_csv(output_path, index=False)
    
    print(f"✓ {ticker}: {len(ticker_df)} records → {output_path}")

print("\n" + "=" * 80)
print("✓ SPLIT COMPLETE")
print("=" * 80)
print("\nYou now have:")
for ticker in tickers:
    print(f"  - yfinance_prices_{ticker}.csv")
print(f"  - yfinance_prices.csv (combined - all tickers)")
