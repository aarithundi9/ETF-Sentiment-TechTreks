"""
Sentiment data acquisition module.

This module provides functions to:
- Fetch sentiment data from various sources (news, Reddit, Twitter)
- Calculate sentiment scores using VADER
- Aggregate sentiment metrics
- Abstract sentiment sources for easy swapping

For now, includes skeleton implementations ready for real API integration.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    TICKERS,
    START_DATE,
    END_DATE,
    SENTIMENT_CONFIG,
    NEWS_API_KEY,
    REDDIT_CLIENT_ID,
    TWITTER_BEARER_TOKEN,
)


class SentimentDataFetcher:
    """
    Abstraction layer for fetching sentiment data.
    
    This class can be configured to use different sentiment sources:
    - Mock data (default for development)
    - News APIs (NewsAPI, etc.)
    - Reddit (via PRAW)
    - Twitter/X
    - CSV files
    """
    
    def __init__(self, source: str = "mock", sentiment_analyzer: str = "vader"):
        """
        Initialize the sentiment fetcher.
        
        Args:
            source: Data source type ('mock', 'real', 'news', 'reddit', 'twitter', 'csv')
                   'real' loads from data/raw/gdelt_sentiment_{ticker}.csv
            sentiment_analyzer: Sentiment analysis tool ('vader' or 'textblob')
        """
        self.source = source
        self.sentiment_analyzer = sentiment_analyzer
        
        if sentiment_analyzer == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                print("Warning: vaderSentiment not installed. Sentiment analysis disabled.")
                self.analyzer = None
        else:
            raise NotImplementedError(f"Sentiment analyzer {sentiment_analyzer} not implemented")
    
    def fetch_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for specified tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date string
            end_date: End date string
            
        Returns:
            DataFrame with sentiment data
        """
        tickers = tickers or TICKERS
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        
        if self.source == "mock":
            return self._fetch_mock_sentiment()
        elif self.source == "real":
            return self._fetch_real_sentiment(tickers, start_date, end_date)
        elif self.source == "news":
            return self._fetch_news_sentiment(tickers, start_date, end_date)
        elif self.source == "reddit":
            return self._fetch_reddit_sentiment(tickers, start_date, end_date)
        elif self.source == "twitter":
            return self._fetch_twitter_sentiment(tickers, start_date, end_date)
        elif self.source == "csv":
            return self._fetch_csv_sentiment()
        else:
            raise ValueError(f"Unknown sentiment source: {self.source}")
    
    def _fetch_mock_sentiment(self) -> pd.DataFrame:
        """Load mock sentiment data from CSV."""
        from src.data.mock_data_generator import load_mock_data
        _, sentiment_df = load_mock_data()
        return sentiment_df
    
    def _fetch_real_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load real sentiment data from GDELT CSV files."""
        from src.config.settings import RAW_DATA_DIR
        
        all_data = []
        for ticker in tickers:
            csv_path = RAW_DATA_DIR / f"gdelt_sentiment_{ticker}.csv"
            if not csv_path.exists():
                print(f"Warning: GDELT sentiment not found for {ticker} at {csv_path}")
                print(f"  Run: python collect_real_data.py --ticker {ticker} --no-price")
                continue
            
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            all_data.append(df)
        
        if not all_data:
            raise FileNotFoundError(
                f"No GDELT sentiment data found for any ticker: {tickers}"
            )
        
        return pd.concat(all_data, ignore_index=True)
    
    def _fetch_news_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch sentiment from news APIs (skeleton for future use).
        
        Example using NewsAPI:
        """
        # from newsapi import NewsApiClient
        # 
        # if not NEWS_API_KEY:
        #     raise ValueError("NEWS_API_KEY not set in environment")
        # 
        # newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # 
        # all_sentiment = []
        # for ticker in tickers:
        #     # Search for ticker-related news
        #     articles = newsapi.get_everything(
        #         q=f"{ticker} OR ETF",
        #         from_param=start_date,
        #         to=end_date,
        #         language='en',
        #         sort_by='relevancy'
        #     )
        #     
        #     # Analyze sentiment for each article
        #     for article in articles['articles']:
        #         text = article['title'] + ' ' + (article['description'] or '')
        #         sentiment_score = self.analyze_text(text)
        #         # ... aggregate by date
        # 
        # return pd.DataFrame(all_sentiment)
        
        raise NotImplementedError(
            "News API integration is ready but not implemented. "
            "Add your API key and uncomment the code above."
        )
    
    def _fetch_reddit_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch sentiment from Reddit (skeleton for future use).
        
        Example using PRAW:
        """
        # import praw
        # 
        # if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        #     raise ValueError("Reddit credentials not set")
        # 
        # reddit = praw.Reddit(
        #     client_id=REDDIT_CLIENT_ID,
        #     client_secret=REDDIT_CLIENT_SECRET,
        #     user_agent=REDDIT_USER_AGENT
        # )
        # 
        # all_sentiment = []
        # for ticker in tickers:
        #     # Search relevant subreddits
        #     subreddit = reddit.subreddit('wallstreetbets+stocks+investing')
        #     for submission in subreddit.search(ticker, time_filter='year'):
        #         text = submission.title + ' ' + submission.selftext
        #         sentiment_score = self.analyze_text(text)
        #         # ... aggregate by date
        # 
        # return pd.DataFrame(all_sentiment)
        
        raise NotImplementedError(
            "Reddit API integration is ready but not implemented. "
            "Add your credentials and uncomment the code above."
        )
    
    def _fetch_twitter_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch sentiment from Twitter/X (skeleton for future use).
        """
        raise NotImplementedError(
            "Twitter API integration skeleton not yet implemented. "
            "Use tweepy or similar library."
        )
    
    def _fetch_csv_sentiment(self) -> pd.DataFrame:
        """Load sentiment from user-provided CSV files."""
        from src.data.user_data_loader import load_user_sentiment_data
        return load_user_sentiment_data()
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        if not self.analyzer:
            return 0.0
        
        if self.sentiment_analyzer == "vader":
            scores = self.analyzer.polarity_scores(text)
            # Compound score is already normalized to [-1, 1]
            return scores['compound']
        else:
            return 0.0


def aggregate_sentiment_by_day(
    sentiment_df: pd.DataFrame,
    min_articles: int = 1,
) -> pd.DataFrame:
    """
    Aggregate raw sentiment data by day and ticker.
    
    Args:
        sentiment_df: DataFrame with raw sentiment (may have multiple records per day)
        min_articles: Minimum number of articles required for valid sentiment
        
    Returns:
        DataFrame with daily aggregated sentiment
    """
    # Group by date and ticker
    daily_sentiment = sentiment_df.groupby(["date", "ticker"]).agg({
        "sentiment_score": "mean",
        "news_count": "sum",
    }).reset_index()
    
    # Filter out days with too few articles
    daily_sentiment = daily_sentiment[
        daily_sentiment["news_count"] >= min_articles
    ].copy()
    
    return daily_sentiment


def calculate_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional sentiment-based features.
    
    Args:
        df: DataFrame with daily sentiment scores
        
    Returns:
        DataFrame with added sentiment features
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Rolling sentiment metrics
    for window in [3, 7, 14]:
        df[f"sentiment_ma_{window}"] = df.groupby("ticker")["sentiment_score"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Sentiment momentum (change in sentiment)
    df["sentiment_change"] = df.groupby("ticker")["sentiment_score"].diff()
    
    # News volume features
    df["news_count_ma_7"] = df.groupby("ticker")["news_count"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Sentiment volatility
    df["sentiment_volatility"] = df.groupby("ticker")["sentiment_score"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    return df


if __name__ == "__main__":
    # Test the sentiment data fetcher
    print("=" * 80)
    print("Testing Sentiment Data Fetcher")
    print("=" * 80)
    
    # Fetch mock sentiment
    fetcher = SentimentDataFetcher(source="mock")
    df = fetcher.fetch_sentiment()
    
    print(f"\nFetched {len(df)} rows of sentiment data")
    print(f"Tickers: {df['ticker'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sentiment range: {df['sentiment_score'].min():.2f} to {df['sentiment_score'].max():.2f}")
    
    # Calculate sentiment features
    df_with_features = calculate_sentiment_features(df)
    
    print("\n" + "=" * 80)
    print("Sample data with sentiment features:")
    print("=" * 80)
    print(df_with_features.head(10))
    
    print("\n" + "=" * 80)
    print("Column names:")
    print("=" * 80)
    print(df_with_features.columns.tolist())
