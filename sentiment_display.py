"""
Module for displaying tweets with their sentiment labels
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .config import RESULTS_DIR


def display_tweets_with_sentiment(merged_data, n_samples=20, save_to_file=True):
    """
    Display tweets with their sentiment labels (positive, neutral, negative).
    
    Args:
        merged_data: Dataframe with tweets and sentiment labels
        n_samples: Number of sample tweets to display
        save_to_file: Whether to save to a text file
    
    Returns:
        DataFrame with tweets and sentiment labels
    """
    # Get tweet column name
    tweet_col = 'tweet' if 'tweet' in merged_data.columns else 'Tweet'
    
    # Select relevant columns
    display_cols = []
    if 'date' in merged_data.columns:
        display_cols.append('date')
    if 'stock name' in merged_data.columns:
        display_cols.append('stock name')
    display_cols.append(tweet_col)
    
    # Add sentiment labels if they exist
    if 'tb_label' in merged_data.columns:
        display_cols.append('tb_label')
        display_cols.append('tb_polarity')
    if 'vader_label' in merged_data.columns:
        display_cols.append('vader_label')
        display_cols.append('vader_compound')
    
    # Create display dataframe
    display_df = merged_data[display_cols].copy()
    
    # Rename columns for better readability
    display_df = display_df.rename(columns={
        tweet_col: 'Tweet',
        'tb_label': 'TextBlob_Sentiment',
        'tb_polarity': 'TextBlob_Score',
        'vader_label': 'VADER_Sentiment',
        'vader_compound': 'VADER_Score'
    })
    
    print("\n" + "="*80)
    print("TWEETS WITH SENTIMENT LABELS")
    print("="*80)
    print(f"\nShowing {min(n_samples, len(display_df))} sample tweets:\n")
    
    # Display sample tweets
    for idx, row in tqdm(display_df.head(n_samples).iterrows(), 
                        total=min(n_samples, len(display_df)), 
                        desc="Displaying tweets"):
        print(f"\n{'─'*80}")
        print(f"Tweet #{idx + 1}")
        if 'date' in row and pd.notna(row['date']):
            print(f"Date: {row['date']}")
        if 'stock name' in row and pd.notna(row['stock name']):
            print(f"Stock: {row['stock name']}")
        print(f"\nTweet: {row['Tweet']}")
        print(f"\nSentiment Labels:")
        if 'TextBlob_Sentiment' in row and pd.notna(row['TextBlob_Sentiment']):
            score = row.get('TextBlob_Score', 'N/A')
            print(f"  📊 TextBlob: {row['TextBlob_Sentiment'].upper():8s} (score: {score:.3f})")
        if 'VADER_Sentiment' in row and pd.notna(row['VADER_Sentiment']):
            score = row.get('VADER_Score', 'N/A')
            print(f"  📊 VADER:    {row['VADER_Sentiment'].upper():8s} (score: {score:.3f})")
    
    print(f"\n{'─'*80}")
    print(f"\nTotal tweets in dataset: {len(display_df):,}")
    
    # Show sentiment distribution
    if 'TextBlob_Sentiment' in display_df.columns:
        print(f"\n📊 TextBlob Sentiment Distribution:")
        tb_dist = display_df['TextBlob_Sentiment'].value_counts()
        for sentiment, count in tb_dist.items():
            pct = (count / len(display_df)) * 100
            print(f"   {sentiment.upper():8s}: {count:6,} ({pct:5.2f}%)")
    
    if 'VADER_Sentiment' in display_df.columns:
        print(f"\n📊 VADER Sentiment Distribution:")
        vader_dist = display_df['VADER_Sentiment'].value_counts()
        for sentiment, count in vader_dist.items():
            pct = (count / len(display_df)) * 100
            print(f"   {sentiment.upper():8s}: {count:6,} ({pct:5.2f}%)")
    
    # Save to file if requested
    if save_to_file:
        output_file = RESULTS_DIR / 'tweets_with_sentiment_labels.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TWEETS WITH SENTIMENT LABELS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total tweets: {len(display_df):,}\n\n")
            
            # Write sentiment distributions
            if 'TextBlob_Sentiment' in display_df.columns:
                f.write("TextBlob Sentiment Distribution:\n")
                tb_dist = display_df['TextBlob_Sentiment'].value_counts()
                for sentiment, count in tb_dist.items():
                    pct = (count / len(display_df)) * 100
                    f.write(f"  {sentiment.upper():8s}: {count:6,} ({pct:5.2f}%)\n")
                f.write("\n")
            
            if 'VADER_Sentiment' in display_df.columns:
                f.write("VADER Sentiment Distribution:\n")
                vader_dist = display_df['VADER_Sentiment'].value_counts()
                for sentiment, count in vader_dist.items():
                    pct = (count / len(display_df)) * 100
                    f.write(f"  {sentiment.upper():8s}: {count:6,} ({pct:5.2f}%)\n")
                f.write("\n")
            
            # Write sample tweets
            f.write("\n" + "="*80 + "\n")
            f.write(f"SAMPLE TWEETS (showing {min(n_samples, len(display_df))} samples)\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in display_df.head(n_samples).iterrows():
                f.write(f"\n{'─'*80}\n")
                f.write(f"Tweet #{idx + 1}\n")
                if 'date' in row and pd.notna(row['date']):
                    f.write(f"Date: {row['date']}\n")
                if 'stock name' in row and pd.notna(row['stock name']):
                    f.write(f"Stock: {row['stock name']}\n")
                f.write(f"\nTweet: {row['Tweet']}\n")
                f.write(f"\nSentiment Labels:\n")
                if 'TextBlob_Sentiment' in row and pd.notna(row['TextBlob_Sentiment']):
                    score = row.get('TextBlob_Score', 'N/A')
                    f.write(f"  TextBlob: {row['TextBlob_Sentiment'].upper():8s} (score: {score:.3f})\n")
                if 'VADER_Sentiment' in row and pd.notna(row['VADER_Sentiment']):
                    score = row.get('VADER_Score', 'N/A')
                    f.write(f"  VADER:    {row['VADER_Sentiment'].upper():8s} (score: {score:.3f})\n")
        
        print(f"\n✅ Saved tweets with sentiment labels to: {output_file}")
    
    return display_df


def save_tweets_with_sentiment_csv(merged_data, output_file=None):
    """
    Save all tweets with their sentiment labels to a CSV file.
    
    Args:
        merged_data: Dataframe with tweets and sentiment labels
        output_file: Path to output file (default: tweets_with_sentiment.csv)
    
    Returns:
        Path to saved file
    """
    if output_file is None:
        output_file = RESULTS_DIR / 'tweets_with_sentiment.csv'
    
    # Get tweet column name
    tweet_col = 'tweet' if 'tweet' in merged_data.columns else 'Tweet'
    
    # Select columns to save
    save_cols = []
    if 'date' in merged_data.columns:
        save_cols.append('date')
    if 'stock name' in merged_data.columns:
        save_cols.append('stock name')
    save_cols.append(tweet_col)
    
    # Add sentiment information
    if 'tb_label' in merged_data.columns:
        save_cols.append('tb_label')
        save_cols.append('tb_polarity')
    if 'vader_label' in merged_data.columns:
        save_cols.append('vader_label')
        save_cols.append('vader_compound')
    
    # Create output dataframe
    output_df = merged_data[save_cols].copy()
    
    # Rename for clarity
    output_df = output_df.rename(columns={
        tweet_col: 'tweet',
        'tb_label': 'textblob_sentiment',
        'tb_polarity': 'textblob_score',
        'vader_label': 'vader_sentiment',
        'vader_compound': 'vader_score'
    })
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(output_df):,} tweets with sentiment labels to: {output_file}")
    
    return output_file

