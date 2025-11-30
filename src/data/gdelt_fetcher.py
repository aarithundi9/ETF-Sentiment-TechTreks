"""
GDELT news sentiment data fetcher.

GDELT (Global Database of Events, Language, and Tone) provides news articles
and sentiment analysis from sources worldwide.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config.settings import RAW_DATA_DIR, GDELT_BASE_URL
import time


class GDELTFetcher:
    """Fetch news and sentiment data from GDELT."""
    
    def __init__(self):
        self.base_url = GDELT_BASE_URL
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def fetch_news_for_date_range(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250,
    ) -> pd.DataFrame:
        """
        Fetch news articles from GDELT for a date range.
        
        Args:
            query: Search query (e.g., "QQQ OR Nasdaq-100")
            start_date: Start date
            end_date: End date
            max_records: Max articles per request (GDELT limit is 250)
            
        Returns:
            DataFrame with news articles and sentiment
        """
        all_articles = []
        current_date = start_date
        
        # GDELT only allows queries in smaller chunks, so we'll do day-by-day
        while current_date <= end_date:
            try:
                print(f"  DEBUG: Fetching {current_date.date()}, query={query[:50]}")  # Debug
                articles = self._fetch_day(query, current_date, max_records)
                print(f"  DEBUG: Got {len(articles)} articles")  # Debug
                if articles:
                    all_articles.extend(articles)
                    print(f"  {current_date.date()}: {len(articles)} articles")
                
                # Rate limiting - be nice to GDELT
                time.sleep(0.5)
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                print(f"  ⚠️  Error fetching {current_date.date()}: {e}")
                import traceback
                traceback.print_exc()  # Debug: show full error
                current_date += timedelta(days=1)
                continue
        
        if not all_articles:
            print("⚠️  No articles found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        return df
    
    def _fetch_day(
        self,
        query: str,
        date: datetime,
        max_records: int = 250,
    ) -> List[dict]:
        """
        Fetch news for a single day from GDELT.
        
        GDELT API format:
        https://api.gdeltproject.org/api/v2/doc/doc?query=QUERY&mode=artlist&maxrecords=250&format=json&startdatetime=YYYYMMDDHHMMSS&enddatetime=YYYYMMDDHHMMSS
        """
        # Format dates for GDELT (YYYYMMDDHHMMSS)
        start_dt = date.replace(hour=0, minute=0, second=0)
        end_dt = date.replace(hour=23, minute=59, second=59)
        
        start_str = start_dt.strftime("%Y%m%d%H%M%S")
        end_str = end_dt.strftime("%Y%m%d%H%M%S")
        
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Check if we got valid content
            if not response.text or len(response.text) < 10:
                return []
            
            try:
                data = response.json()
            except ValueError as e:
                # JSON parsing failed
                return []
            
            if "articles" not in data:
                return []
            
            articles = []
            for article in data["articles"]:
                # Calculate sentiment using VADER
                title = article.get("title", "")
                sentiment = self.sentiment_analyzer.polarity_scores(title)
                
                articles.append({
                    "date": date.date(),
                    "title": title,
                    "url": article.get("url", ""),
                    "domain": article.get("domain", ""),
                    "seendate": article.get("seendate", ""),
                    "sentiment_score": sentiment["compound"],
                    "sentiment_positive": sentiment["pos"],
                    "sentiment_negative": sentiment["neg"],
                    "sentiment_neutral": sentiment["neu"],
                })
            
            return articles
            
        except Exception as e:
            raise Exception(f"GDELT API error: {e}")
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate article-level sentiment to daily sentiment scores.
        
        Args:
            df: DataFrame with individual articles
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if df.empty:
            return pd.DataFrame()
        
        daily = df.groupby("date").agg({
            "sentiment_score": ["mean", "std", "min", "max"],
            "sentiment_positive": "mean",
            "sentiment_negative": "mean",
            "sentiment_neutral": "mean",
            "title": "count",  # Number of articles
        }).reset_index()
        
        # Flatten column names
        daily.columns = [
            "date",
            "sentiment_score",
            "sentiment_std",
            "sentiment_min",
            "sentiment_max",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "news_count",
        ]
        
        return daily


def fetch_gdelt_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch and aggregate GDELT news sentiment for a ticker.
    
    Args:
        ticker: Ticker symbol (e.g., "QQQ")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        save: Whether to save to CSV
        
    Returns:
        DataFrame with daily sentiment scores
    """
    # Create search query
    # For QQQ: search for "QQQ" (simpler query works better with GDELT)
    query_map = {
        "QQQ": "QQQ",
        "SPY": "SPY",
        "IWM": "IWM",
    }
    
    query = query_map.get(ticker, ticker)
    
    print(f"Fetching GDELT news for {ticker}")
    print(f"  Query: {query}")
    print(f"  Date range: {start_date} to {end_date}")
    
    fetcher = GDELTFetcher()
    
    # Convert dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Fetch articles
    print("\nFetching articles...")
    articles_df = fetcher.fetch_news_for_date_range(query, start_dt, end_dt)
    
    if articles_df.empty:
        print("⚠️  No articles found")
        return pd.DataFrame()
    
    # Save raw articles
    if save:
        articles_path = RAW_DATA_DIR / f"gdelt_articles_{ticker}.csv"
        articles_df.to_csv(articles_path, index=False)
        print(f"\n✓ Saved {len(articles_df)} articles to {articles_path}")
    
    # Aggregate to daily sentiment
    print("\nAggregating to daily sentiment...")
    daily_df = fetcher.aggregate_daily_sentiment(articles_df)
    daily_df["ticker"] = ticker
    
    # Reorder columns
    daily_df = daily_df[["date", "ticker", "sentiment_score", "sentiment_std",
                         "sentiment_min", "sentiment_max", "sentiment_positive",
                         "sentiment_negative", "sentiment_neutral", "news_count"]]
    
    # Save daily sentiment
    if save:
        daily_path = RAW_DATA_DIR / f"gdelt_sentiment_{ticker}.csv"
        daily_df.to_csv(daily_path, index=False)
        print(f"✓ Saved {len(daily_df)} days to {daily_path}")
    
    print(f"\n✓ Sentiment stats:")
    print(f"  Avg sentiment: {daily_df['sentiment_score'].mean():.3f}")
    print(f"  Avg articles/day: {daily_df['news_count'].mean():.1f}")
    print(f"  Total articles: {daily_df['news_count'].sum()}")
    
    return daily_df


if __name__ == "__main__":
    # Test GDELT fetcher
    print("=" * 80)
    print("Testing GDELT Fetcher")
    print("=" * 80)
    
    # Test with just a few days
    df = fetch_gdelt_sentiment(
        ticker="QQQ",
        start_date="2024-11-01",
        end_date="2024-11-07",
        save=False,
    )
    
    if not df.empty:
        print("\nSample data:")
        print(df.head())
