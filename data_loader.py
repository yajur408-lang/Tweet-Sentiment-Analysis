"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .config import TWEETS_CSV, PRICES_CSV
from .utils import normalize_column_names


def load_data(tweets_path=None, prices_path=None):
    """
    Load stock tweets and price data from CSV files.
    
    Args:
        tweets_path: Path to tweets CSV file
        prices_path: Path to prices CSV file
    
    Returns:
        tuple: (tweets_df, prices_df)
    """
    tweets_path = tweets_path or TWEETS_CSV
    prices_path = prices_path or PRICES_CSV
    
    print("Loading data...")
    tweets = pd.read_csv(tweets_path)
    prices = pd.read_csv(prices_path)
    
    print(f"[OK] Loaded {len(tweets):,} tweets and {len(prices):,} price records")
    return tweets, prices


def preprocess_data(tweets, prices):
    """
    Preprocess data: remove duplicates, convert dates, handle missing values.
    
    Args:
        tweets: Tweets dataframe
        prices: Prices dataframe
    
    Returns:
        tuple: (cleaned_tweets, cleaned_prices)
    """
    # Drop duplicates
    tweets = tweets.drop_duplicates()
    prices = prices.drop_duplicates()
    
    # Convert dates
    tweets['Date'] = pd.to_datetime(tweets['Date'], errors='coerce')
    prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    tweets = tweets.dropna(subset=['Date'])
    prices = prices.dropna(subset=['Date'])
    
    # Normalize column names
    tweets = normalize_column_names(tweets)
    prices = normalize_column_names(prices)
    
    print(f"[OK] Preprocessed data: {len(tweets):,} tweets, {len(prices):,} price records")
    return tweets, prices


def merge_datasets(tweets, prices):
    """
    Merge tweets and prices datasets on Date and Stock Name.
    
    Args:
        tweets: Tweets dataframe
        prices: Prices dataframe
    
    Returns:
        Merged dataframe
    """
    # Handle both original and normalized column names
    date_col = 'date' if 'date' in tweets.columns else 'Date'
    stock_col = 'stock name' if 'stock name' in tweets.columns else 'Stock Name'
    
    # Normalize datetime columns (remove timezone if present)
    if date_col in tweets.columns:
        tweets[date_col] = pd.to_datetime(tweets[date_col]).dt.tz_localize(None)
    if date_col in prices.columns:
        prices[date_col] = pd.to_datetime(prices[date_col]).dt.tz_localize(None)
    
    # Also normalize to date only (remove time component) for better matching
    tweets[date_col] = pd.to_datetime(tweets[date_col]).dt.date
    prices[date_col] = pd.to_datetime(prices[date_col]).dt.date
    
    merged = pd.merge(tweets, prices, on=[date_col, stock_col], how='inner')
    print(f"[OK] Merged dataset shape: {merged.shape}")
    return merged


def create_target_variable(merged, close_col=None):
    """
    Create target variable (next day price change direction).
    
    Args:
        merged: Merged dataframe
        close_col: Name of closing price column
    
    Returns:
        Dataframe with target variable
    """
    from .utils import detect_close_column
    
    if close_col is None:
        close_col = detect_close_column(merged)
    
    # Sort by stock and date
    merged = merged.sort_values(['stock name', 'date']).reset_index(drop=True)
    
    # Create next-day price change
    merged['next_day_close'] = merged.groupby('stock name')[close_col].shift(-1)
    merged['next_day_change'] = merged['next_day_close'] - merged[close_col]
    merged['target'] = (merged['next_day_change'] > 0).astype(int)
    
    # Drop NaN rows created by shifting
    merged = merged.dropna(subset=['next_day_change'])
    
    print(f"[OK] Target variable created! {len(merged):,} records with target")
    return merged


def get_data_summary(tweets, prices):
    """Print summary statistics of the datasets."""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    # Handle both original and normalized column names
    date_col = 'date' if 'date' in tweets.columns else 'Date'
    stock_col = 'stock name' if 'stock name' in tweets.columns else 'Stock Name'
    
    print(f"\nTweets dataset:")
    print(f"  - Total tweets: {len(tweets):,}")
    if date_col in tweets.columns:
        print(f"  - Date range: {tweets[date_col].min()} to {tweets[date_col].max()}")
    if stock_col in tweets.columns:
        print(f"  - Unique stocks: {tweets[stock_col].nunique()}")
        print(f"  - Stocks: {tweets[stock_col].unique().tolist()}")
    
    print(f"\nPrices dataset:")
    print(f"  - Total records: {len(prices):,}")
    if date_col in prices.columns:
        print(f"  - Date range: {prices[date_col].min()} to {prices[date_col].max()}")
    if stock_col in prices.columns:
        print(f"  - Unique stocks: {prices[stock_col].nunique()}")
    print(f"  - Missing values: {prices.isnull().sum().sum()}")

