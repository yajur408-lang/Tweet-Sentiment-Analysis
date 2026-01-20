"""
Sentiment analysis module using TextBlob and VADER
"""
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()
from .utils import blob_sentiment, vader_label


def compute_textblob_sentiment(merged):
    """
    Compute TextBlob sentiment scores (polarity and subjectivity).
    
    Args:
        merged: Dataframe with 'Tweet' or 'tweet' column
    
    Returns:
        Dataframe with TextBlob sentiment features
    """
    print("Computing TextBlob sentiment...")
    tweet_col = 'tweet' if 'tweet' in merged.columns else 'Tweet'
    
    # Use tqdm for progress tracking
    merged['tb_polarity'] = merged[tweet_col].progress_apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    merged['tb_subjectivity'] = merged[tweet_col].progress_apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )
    merged['tb_label'] = merged['tb_polarity'].progress_apply(blob_sentiment)
    print("[OK] TextBlob sentiment computed")
    return merged


def compute_vader_sentiment(merged):
    """
    Compute VADER sentiment scores.
    
    Args:
        merged: Dataframe with 'Tweet' or 'tweet' column
    
    Returns:
        Dataframe with VADER sentiment features
    """
    print("Computing VADER sentiment...")
    tweet_col = 'tweet' if 'tweet' in merged.columns else 'Tweet'
    analyzer = SentimentIntensityAnalyzer()
    merged['vader_compound'] = merged[tweet_col].progress_apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    merged['vader_label'] = merged['vader_compound'].progress_apply(vader_label)
    print("[OK] VADER sentiment computed")
    return merged


def compute_text_features(merged, finance_keywords=None):
    """
    Compute text structure metrics and finance keyword counts.
    
    IMPORTANT: All features use only the tweet text itself, which is available
    at prediction time. No target variable or future data is used.
    
    Args:
        merged: Dataframe with 'Tweet' or 'tweet' column
        finance_keywords: List of finance-related keywords to count
    
    Returns:
        Dataframe with text features
    """
    from .config import FINANCE_KEYWORDS
    
    if finance_keywords is None:
        finance_keywords = FINANCE_KEYWORDS
    
    print("Computing text features (using only tweet text)...")
    
    tweet_col = 'tweet' if 'tweet' in merged.columns else 'Tweet'
    
    # Basic text metrics - all derived from tweet text only
    merged['tweet_length'] = merged[tweet_col].progress_apply(lambda x: len(str(x)))
    merged['word_count'] = merged[tweet_col].progress_apply(lambda x: len(str(x).split()))
    merged['hashtag_count'] = merged[tweet_col].progress_apply(lambda x: str(x).count('#'))
    merged['has_mention'] = merged[tweet_col].progress_apply(lambda x: '@' in str(x))
    merged['has_url'] = merged[tweet_col].progress_apply(lambda x: 'http' in str(x))
    
    # Finance keyword counts - derived from tweet text only
    for term in tqdm(finance_keywords, desc="Processing finance keywords"):
        merged[f'count_{term}'] = merged[tweet_col].str.lower().str.count(term)
    
    print("[OK] Text features computed")
    print("  [SECURE] All features use only data available at prediction time (tweet text)")
    return merged


def compute_all_sentiment_features(merged, finance_keywords=None):
    """
    Compute all sentiment and text features.
    
    Args:
        merged: Dataframe with 'Tweet' column
        finance_keywords: List of finance-related keywords
    
    Returns:
        Dataframe with all sentiment and text features
    """
    merged = compute_textblob_sentiment(merged)
    merged = compute_vader_sentiment(merged)
    merged = compute_text_features(merged, finance_keywords)
    return merged

