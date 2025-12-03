"""Collect price data for multiple tickers and save to one CSV."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.yfinance_fetcher import fetch_yfinance_data
import pandas as pd

# Tickers to collect
TICKERS = ["XLY", "XLI", "XLF"]
START_DATE = "2021-01-01"
END_DATE = "2025-10-31"

print("=" * 80)
print("COLLECTING PRICE DATA FOR MULTIPLE TICKERS")
print("=" * 80)
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Date range: {START_DATE} to {END_DATE}")
print()

# Collect data for all tickers
all_data = []
for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    df = fetch_yfinance_data([ticker], START_DATE, END_DATE)
    if df is not None and len(df) > 0:
        all_data.append(df)
        print(f"  ✓ {len(df)} records for {ticker}")
    else:
        print(f"  ⚠ No data for {ticker}")

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save
    output_path = Path("data/raw/yfinance_prices.csv")
    combined_df.to_csv(output_path, index=False)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(combined_df)}")
    print("\nRecords per ticker:")
    print(combined_df['ticker'].value_counts())
    print(f"\n✓ Saved to: {output_path}")
else:
    print("\n❌ No data collected")
