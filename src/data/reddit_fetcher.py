"""
Reddit sentiment data fetcher.

Fetches posts and comments from investing/trading subreddits
and analyzes sentiment.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config.settings import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    RAW_DATA_DIR,
)
import time


class RedditFetcher:
    """Fetch posts and sentiment from Reddit."""
    
    def __init__(self):
        """Initialize Reddit API client."""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            raise ValueError(
                "Reddit API credentials not found. "
                "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file"
            )
        
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def fetch_posts(
        self,
        query: str,
        subreddits: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch Reddit posts matching query from specified subreddits.
        
        Args:
            query: Search query (e.g., "QQQ")
            subreddits: List of subreddit names
            start_date: Start date
            end_date: End date
            limit: Max posts per subreddit
            
        Returns:
            DataFrame with posts and sentiment
        """
        all_posts = []
        
        for subreddit_name in subreddits:
            print(f"  Searching r/{subreddit_name}...")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts
                for post in subreddit.search(query, limit=limit):
                    post_date = datetime.fromtimestamp(post.created_utc)
                    
                    # Filter by date range
                    if start_date <= post_date <= end_date:
                        # Combine title and selftext for sentiment
                        text = f"{post.title} {post.selftext}"
                        sentiment = self.sentiment_analyzer.polarity_scores(text)
                        
                        all_posts.append({
                            "date": post_date.date(),
                            "subreddit": subreddit_name,
                            "title": post.title,
                            "text": post.selftext[:500],  # Truncate long posts
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "url": post.url,
                            "sentiment_score": sentiment["compound"],
                            "sentiment_positive": sentiment["pos"],
                            "sentiment_negative": sentiment["neg"],
                            "sentiment_neutral": sentiment["neu"],
                        })
                
                print(f"    Found {len([p for p in all_posts if p['subreddit'] == subreddit_name])} posts")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                continue
        
        if not all_posts:
            print("  ⚠️  No posts found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_posts)
        return df
    
    def aggregate_daily_sentiment(
        self,
        df: pd.DataFrame,
        weight_by_score: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate post-level sentiment to daily sentiment scores.
        
        Args:
            df: DataFrame with individual posts
            weight_by_score: Weight sentiment by post score (upvotes)
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if df.empty:
            return pd.DataFrame()
        
        if weight_by_score:
            # Weight sentiment by post score (more upvoted = more influential)
            df["weighted_sentiment"] = df["sentiment_score"] * df["score"]
            
            daily = df.groupby("date").agg({
                "weighted_sentiment": "sum",
                "score": "sum",
                "sentiment_score": ["mean", "std"],
                "num_comments": "sum",
                "title": "count",
            }).reset_index()
            
            # Calculate weighted average
            daily["sentiment_score"] = daily["weighted_sentiment"] / daily["score"]
            daily = daily.drop("weighted_sentiment", axis=1)
            
        else:
            daily = df.groupby("date").agg({
                "sentiment_score": ["mean", "std"],
                "score": "sum",
                "num_comments": "sum",
                "title": "count",
            }).reset_index()
        
        # Flatten column names
        daily.columns = [
            "date",
            "total_score",
            "sentiment_score",
            "sentiment_std",
            "total_comments",
            "post_count",
        ]
        
        return daily


def fetch_reddit_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    subreddits: Optional[List[str]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch and aggregate Reddit sentiment for a ticker.
    
    Args:
        ticker: Ticker symbol (e.g., "QQQ")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        subreddits: List of subreddits to search (default: investing subreddits)
        save: Whether to save to CSV
        
    Returns:
        DataFrame with daily sentiment scores
    """
    # Default subreddits for ETF/market discussion
    if subreddits is None:
        subreddits = [
            "wallstreetbets",
            "stocks",
            "investing",
            "StockMarket",
            "options",
            "Daytrading",
        ]
    
    print(f"Fetching Reddit sentiment for {ticker}")
    print(f"  Subreddits: {', '.join(subreddits)}")
    print(f"  Date range: {start_date} to {end_date}")
    
    fetcher = RedditFetcher()
    
    # Convert dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Fetch posts
    print("\nFetching posts...")
    posts_df = fetcher.fetch_posts(
        query=ticker,
        subreddits=subreddits,
        start_date=start_dt,
        end_date=end_dt,
    )
    
    if posts_df.empty:
        print("⚠️  No posts found")
        return pd.DataFrame()
    
    # Save raw posts
    if save:
        posts_path = RAW_DATA_DIR / f"reddit_posts_{ticker}.csv"
        posts_df.to_csv(posts_path, index=False)
        print(f"\n✓ Saved {len(posts_df)} posts to {posts_path}")
    
    # Aggregate to daily sentiment
    print("\nAggregating to daily sentiment...")
    daily_df = fetcher.aggregate_daily_sentiment(posts_df, weight_by_score=True)
    daily_df["ticker"] = ticker
    
    # Reorder columns
    daily_df = daily_df[["date", "ticker", "sentiment_score", "sentiment_std",
                         "post_count", "total_score", "total_comments"]]
    
    # Save daily sentiment
    if save:
        daily_path = RAW_DATA_DIR / f"reddit_sentiment_{ticker}.csv"
        daily_df.to_csv(daily_path, index=False)
        print(f"✓ Saved {len(daily_df)} days to {daily_path}")
    
    print(f"\n✓ Sentiment stats:")
    print(f"  Avg sentiment: {daily_df['sentiment_score'].mean():.3f}")
    print(f"  Avg posts/day: {daily_df['post_count'].mean():.1f}")
    print(f"  Total posts: {daily_df['post_count'].sum()}")
    
    return daily_df


if __name__ == "__main__":
    # Test Reddit fetcher
    print("=" * 80)
    print("Testing Reddit Fetcher")
    print("=" * 80)
    print("\n⚠️  NOTE: You need Reddit API credentials in .env file:")
    print("  REDDIT_CLIENT_ID=your_client_id")
    print("  REDDIT_CLIENT_SECRET=your_client_secret")
    print("  Get them at: https://www.reddit.com/prefs/apps\n")
    
    try:
        # Test with just a few days
        df = fetch_reddit_sentiment(
            ticker="QQQ",
            start_date="2024-11-01",
            end_date="2024-11-07",
            save=False,
        )
        
        if not df.empty:
            print("\nSample data:")
            print(df.head())
    except ValueError as e:
        print(f"\n❌ {e}")
