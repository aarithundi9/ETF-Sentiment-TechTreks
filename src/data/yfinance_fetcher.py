"""
Real price data fetcher using yfinance.

This module fetches historical OHLCV data for ETFs from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime
from src.config.settings import TICKERS, START_DATE, END_DATE, RAW_DATA_DIR


def fetch_yfinance_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols (default: from config)
        start_date: Start date YYYY-MM-DD (default: from config)
        end_date: End date YYYY-MM-DD (default: from config)
        save: Whether to save to CSV
        
    Returns:
        DataFrame with OHLCV data
    """
    tickers = tickers or TICKERS
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    all_data = []
    
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        
        try:
            # Download data using yfinance
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Adjust for splits/dividends
            )
            
            if df.empty:
                print(f"  ⚠️  No data found for {ticker}")
                continue
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Rename columns to lowercase (handle both string and tuple column names)
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() 
                         for col in df.columns]
            
            # Add ticker column
            df["ticker"] = ticker
            
            # Reorder columns
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            
            all_data.append(df)
            print(f"  ✓ Downloaded {len(df)} days for {ticker}")
            print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"    Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
        except Exception as e:
            print(f"  ❌ Error downloading {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully downloaded for any ticker")
    
    # Combine all tickers
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    if save:
        output_path = RAW_DATA_DIR / "yfinance_prices.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(combined_df)} records to {output_path}")
    
    return combined_df


def get_latest_price(ticker: str) -> dict:
    """
    Get the most recent price data for a ticker.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dict with latest price info
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
        "previous_close": info.get("previousClose"),
        "day_change_pct": info.get("regularMarketChangePercent"),
        "volume": info.get("volume"),
        "market_cap": info.get("marketCap"),
    }


if __name__ == "__main__":
    # Test the fetcher
    print("=" * 80)
    print("Testing yfinance Data Fetcher")
    print("=" * 80)
    
    df = fetch_yfinance_data()
    
    print(f"\n✓ Total records: {len(df)}")
    print(f"\nSample data:")
    print(df.head(10))
    
    print(f"\nData info:")
    print(df.info())
