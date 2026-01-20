"""
Utility functions for Stock Sentiment Analysis
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def normalize_column_names(df):
    """Normalize column names to lowercase with stripped whitespace."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def detect_close_column(df):
    """Detect which column to use for closing price."""
    close_candidates = ['close', 'adj close', 'close price']
    for col in close_candidates:
        if col in df.columns:
            return col
    raise KeyError("No 'close' column found in dataset. Available columns: " + str(df.columns.tolist()))


def blob_sentiment(score):
    """Convert TextBlob polarity score to sentiment label."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"


def vader_label(score):
    """Convert VADER compound score to sentiment label."""
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def print_section(title, width=60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)






