"""
Real data collection script.

This script fetches real price and sentiment data for ETFs.

Usage:
    python collect_real_data.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
"""

import argparse
from datetime import datetime
from src.data.yfinance_fetcher import fetch_yfinance_data
from src.data.gdelt_fetcher import fetch_gdelt_sentiment
from src.data.reddit_fetcher import fetch_reddit_sentiment
from src.config.settings import TICKERS, START_DATE, END_DATE


def collect_all_data(
    ticker: str,
    start_date: str,
    end_date: str,
    include_price: bool = True,
    include_gdelt: bool = True,
    include_reddit: bool = True,
):
    """
    Collect all real data sources for a ticker.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        include_price: Fetch yfinance price data
        include_gdelt: Fetch GDELT news sentiment
        include_reddit: Fetch Reddit sentiment
    """
    print("=" * 80)
    print(f"COLLECTING REAL DATA FOR {ticker}")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}\n")
    
    results = {}
    
    # 1. Fetch price data from yfinance
    if include_price:
        print("\n" + "=" * 80)
        print("1. FETCHING PRICE DATA (yfinance)")
        print("=" * 80)
        try:
            price_df = fetch_yfinance_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                save=True,
            )
            results["price"] = price_df
            print(f"✓ Price data: {len(price_df)} records")
        except Exception as e:
            print(f"❌ Error fetching price data: {e}")
            results["price"] = None
    
    # 2. Fetch GDELT news sentiment
    if include_gdelt:
        print("\n" + "=" * 80)
        print("2. FETCHING NEWS SENTIMENT (GDELT)")
        print("=" * 80)
        print("⚠️  Note: GDELT fetching can take a while for large date ranges")
        print("   Consider starting with a smaller date range first\n")
        try:
            gdelt_df = fetch_gdelt_sentiment(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                save=True,
            )
            results["gdelt"] = gdelt_df
            print(f"✓ GDELT sentiment: {len(gdelt_df)} days")
        except Exception as e:
            print(f"❌ Error fetching GDELT data: {e}")
            results["gdelt"] = None
    
    # 3. Fetch Reddit sentiment
    if include_reddit:
        print("\n" + "=" * 80)
        print("3. FETCHING REDDIT SENTIMENT")
        print("=" * 80)
        print("⚠️  Note: Reddit API requires credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)")
        print("   Get them at: https://www.reddit.com/prefs/apps\n")
        try:
            reddit_df = fetch_reddit_sentiment(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                save=True,
            )
            results["reddit"] = reddit_df
            print(f"✓ Reddit sentiment: {len(reddit_df)} days")
        except Exception as e:
            print(f"❌ Error fetching Reddit data: {e}")
            results["reddit"] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("DATA COLLECTION SUMMARY")
    print("=" * 80)
    for source, df in results.items():
        if df is not None and not df.empty:
            print(f"✓ {source.upper()}: {len(df)} records")
        else:
            print(f"❌ {source.upper()}: No data collected")
    
    print("\n✓ Data saved to data/raw/")
    print("  Next steps:")
    print("  1. Run: python main.py pipeline --use-real-data")
    print("  2. Run: python main.py train --use-real-data")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect real ETF price and sentiment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all data for QQQ from 2020-2024
  python collect_real_data.py --ticker QQQ --start 2020-01-01 --end 2024-12-31
  
  # Collect only price data (fast)
  python collect_real_data.py --ticker QQQ --price-only
  
  # Test with small date range (recommended first)
  python collect_real_data.py --ticker QQQ --start 2024-11-01 --end 2024-11-30
        """
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default=TICKERS[0],
        help=f"Ticker symbol (default: {TICKERS[0]})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=END_DATE,
        help=f"End date YYYY-MM-DD (default: {END_DATE})",
    )
    parser.add_argument(
        "--price-only",
        action="store_true",
        help="Only fetch price data (skip sentiment sources)",
    )
    parser.add_argument(
        "--no-price",
        action="store_true",
        help="Skip price data collection",
    )
    parser.add_argument(
        "--no-gdelt",
        action="store_true",
        help="Skip GDELT news collection",
    )
    parser.add_argument(
        "--no-reddit",
        action="store_true",
        help="Skip Reddit collection",
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("❌ Error: Dates must be in YYYY-MM-DD format")
        return
    
    # Determine what to fetch
    if args.price_only:
        include_gdelt = False
        include_reddit = False
    else:
        include_gdelt = not args.no_gdelt
        include_reddit = not args.no_reddit
    
    # Run collection
    collect_all_data(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        include_price=not args.no_price,
        include_gdelt=include_gdelt,
        include_reddit=include_reddit,
    )


if __name__ == "__main__":
    main()
