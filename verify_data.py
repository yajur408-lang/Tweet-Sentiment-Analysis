"""
Data Verification Script
Loads and verifies tweets_with_sentiment.csv with samples and groupby analysis
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import RESULTS_DIR


def verify_data():
    """Load and verify tweets_with_sentiment.csv"""
    csv_path = RESULTS_DIR / 'tweets_with_sentiment.csv'
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    print("="*80)
    print("DATA VERIFICATION")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df):,} rows")
    print(f"✅ Columns: {', '.join(df.columns.tolist())}")
    
    # Check for required columns
    required_cols = ['tweet', 'textblob_sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE TWEETS (20 random samples)")
    print("="*80)
    
    # Map column names if needed
    display_cols = []
    if 'tweet' in df.columns:
        display_cols.append('tweet')
    elif 'Tweet' in df.columns:
        display_cols.append('Tweet')
    
    if 'textblob_sentiment' in df.columns:
        display_cols.append('textblob_sentiment')
    if 'vader_sentiment' in df.columns:
        display_cols.append('vader_sentiment')
    if 'target' in df.columns:
        display_cols.append('target')
    if 'date' in df.columns:
        display_cols.append('date')
    if 'stock name' in df.columns:
        display_cols.append('stock name')
    
    sample_df = df[display_cols].sample(min(20, len(df)))
    print(sample_df.to_string())
    
    # Group by sentiment label
    print("\n" + "="*80)
    print("GROUP BY SENTIMENT LABEL (3 samples per label)")
    print("="*80)
    
    if 'textblob_sentiment' in df.columns:
        print("\n📊 TextBlob Sentiment:")
        for sentiment in df['textblob_sentiment'].unique():
            if pd.notna(sentiment):
                sentiment_df = df[df['textblob_sentiment'] == sentiment]
                print(f"\n{sentiment.upper()} ({len(sentiment_df):,} total):")
                sample = sentiment_df.head(3)[display_cols]
                print(sample.to_string())
    
    if 'vader_sentiment' in df.columns:
        print("\n📊 VADER Sentiment:")
        for sentiment in df['vader_sentiment'].unique():
            if pd.notna(sentiment):
                sentiment_df = df[df['vader_sentiment'] == sentiment]
                print(f"\n{sentiment.upper()} ({len(sentiment_df):,} total):")
                sample = sentiment_df.head(3)[display_cols]
                print(sample.to_string())
    
    # Verify 50+ samples per sentiment class
    print("\n" + "="*80)
    print("SENTIMENT CLASS VERIFICATION (50+ samples required)")
    print("="*80)
    
    if 'textblob_sentiment' in df.columns:
        print("\n📊 TextBlob Sentiment Distribution:")
        sentiment_counts = df['textblob_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            status = "✅" if count >= 50 else "❌"
            print(f"  {status} {sentiment.upper():10s}: {count:6,} samples")
    
    if 'vader_sentiment' in df.columns:
        print("\n📊 VADER Sentiment Distribution:")
        sentiment_counts = df['vader_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            status = "✅" if count >= 50 else "❌"
            print(f"  {status} {sentiment.upper():10s}: {count:6,} samples")
    
    # Check for target variable
    if 'target' in df.columns:
        print("\n📊 Target Variable Distribution:")
        target_counts = df['target'].value_counts()
        for target, count in target_counts.items():
            print(f"  Target {target}: {count:6,} samples ({count/len(df)*100:.2f}%)")
    else:
        print("\n⚠️  'target' column not found in CSV")
        print("   Note: Target variable may be in merged_clean_with_embeddings.csv")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"\nDate range:")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"  From: {df['date'].min()}")
        print(f"  To: {df['date'].max()}")
    
    if 'stock name' in df.columns:
        print(f"\nUnique stocks: {df['stock name'].nunique()}")
        print(f"Stocks: {', '.join(sorted(df['stock name'].unique().astype(str)))}")
    
    return df


if __name__ == "__main__":
    df = verify_data()
    if df is not None:
        print("\n✅ Data verification complete!")

