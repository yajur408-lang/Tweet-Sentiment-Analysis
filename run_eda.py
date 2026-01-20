"""
Data Exploration and Visualization Script
Run this script to perform EDA and generate all visualization plots
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import (
    load_data, preprocess_data, merge_datasets, 
    create_target_variable, get_data_summary
)
from src.sentiment_analysis import compute_all_sentiment_features
from src.feature_engineering import compute_technical_indicators
from src.visualization import (
    plot_basic_eda, plot_advanced_eda, plot_correlation_matrix,
    plot_sentiment_analysis
)
from src.utils import print_section
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib backend for better compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def main():
    """Main execution function for EDA and visualization."""
    print_section("DATA EXPLORATION AND VISUALIZATION", 60)
    
    # Step 1: Load and preprocess data
    print_section("STEP 1: DATA LOADING AND PREPROCESSING", 60)
    tweets, prices = load_data()
    tweets, prices = preprocess_data(tweets, prices)
    get_data_summary(tweets, prices)
    
    # Step 2: Basic EDA
    print_section("STEP 2: BASIC EXPLORATORY DATA ANALYSIS", 60)
    print("Generating basic EDA plots...")
    try:
        plot_basic_eda(tweets, prices)
        print("[OK] Basic EDA plots saved to results/basic_eda.png")
    except Exception as e:
        print(f"[WARNING] Error creating basic EDA plots: {e}")
    
    # Step 3: Advanced EDA
    print_section("STEP 3: ADVANCED EXPLORATORY DATA ANALYSIS", 60)
    print("Generating advanced EDA plots...")
    try:
        plot_advanced_eda(tweets, prices)
        print("[OK] Advanced EDA plots saved to results/advanced_eda.png")
    except Exception as e:
        print(f"[WARNING] Error creating advanced EDA plots: {e}")
    
    # Step 4: Merge datasets
    print_section("STEP 4: MERGING DATASETS", 60)
    merged = merge_datasets(tweets, prices)
    
    # Step 5: Compute sentiment features
    print_section("STEP 5: COMPUTING SENTIMENT FEATURES", 60)
    merged = compute_all_sentiment_features(merged)
    
    # Step 6: Create target variable
    print_section("STEP 6: CREATING TARGET VARIABLE", 60)
    merged = create_target_variable(merged)
    
    # Step 7: Feature engineering
    print_section("STEP 7: COMPUTING TECHNICAL INDICATORS", 60)
    merged = compute_technical_indicators(merged)
    
    # Step 8: Correlation Analysis
    print_section("STEP 8: CORRELATION ANALYSIS", 60)
    print("Generating correlation heatmap...")
    try:
        plot_correlation_matrix(merged)
        print("[OK] Correlation matrix saved to results/correlation_matrix.png")
    except Exception as e:
        print(f"[WARNING] Error creating correlation matrix: {e}")
    
    # Step 9: Sentiment Analysis Visualizations
    print_section("STEP 9: SENTIMENT ANALYSIS VISUALIZATIONS", 60)
    print("Generating sentiment analysis plots...")
    try:
        plot_sentiment_analysis(merged)
        print("[OK] Sentiment analysis plots saved to results/sentiment_analysis.png")
    except Exception as e:
        print(f"[WARNING] Error creating sentiment plots: {e}")
    
    # Step 10: Additional Statistics
    print_section("STEP 10: ADDITIONAL STATISTICS", 60)
    
    print("\nPrice Statistics:")
    if 'close' in merged.columns:
        print(merged['close'].describe())
    elif 'Close' in merged.columns:
        print(merged['Close'].describe())
    
    print("\nSentiment Statistics:")
    print(f"  - Average TextBlob polarity: {merged['tb_polarity'].mean():.4f}")
    print(f"  - Average VADER compound: {merged['vader_compound'].mean():.4f}")
    print(f"  - Positive tweets: {(merged['tb_label'] == 'positive').sum():,} ({(merged['tb_label'] == 'positive').mean()*100:.1f}%)")
    print(f"  - Negative tweets: {(merged['tb_label'] == 'negative').sum():,} ({(merged['tb_label'] == 'negative').mean()*100:.1f}%)")
    print(f"  - Neutral tweets: {(merged['tb_label'] == 'neutral').sum():,} ({(merged['tb_label'] == 'neutral').mean()*100:.1f}%)")
    
    if 'next_day_change' in merged.columns:
        print("\nTarget Variable Statistics:")
        print(f"  - Average next-day change: ${merged['next_day_change'].mean():.2f}")
        print(f"  - Std dev: ${merged['next_day_change'].std():.2f}")
        print(f"  - Up days: {(merged['target'] == 1).sum():,} ({(merged['target'] == 1).mean()*100:.1f}%)")
        print(f"  - Down days: {(merged['target'] == 0).sum():,} ({(merged['target'] == 0).mean()*100:.1f}%)")
    
    # Step 11: Save summary statistics
    print_section("STEP 11: SAVING SUMMARY STATISTICS", 60)
    from src.config import RESULTS_DIR
    
    # Save descriptive statistics
    stats_file = RESULTS_DIR / "data_summary_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATA SUMMARY STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Dataset Shape:\n")
        f.write(f"  - Total records: {len(merged):,}\n")
        f.write(f"  - Features: {merged.shape[1]}\n\n")
        
        f.write("Sentiment Distribution:\n")
        f.write(f"  - Positive: {(merged['tb_label'] == 'positive').sum():,} ({(merged['tb_label'] == 'positive').mean()*100:.1f}%)\n")
        f.write(f"  - Negative: {(merged['tb_label'] == 'negative').sum():,} ({(merged['tb_label'] == 'negative').mean()*100:.1f}%)\n")
        f.write(f"  - Neutral: {(merged['tb_label'] == 'neutral').sum():,} ({(merged['tb_label'] == 'neutral').mean()*100:.1f}%)\n\n")
        
        if 'next_day_change' in merged.columns:
            f.write("Target Variable:\n")
            f.write(f"  - Up days: {(merged['target'] == 1).sum():,} ({(merged['target'] == 1).mean()*100:.1f}%)\n")
            f.write(f"  - Down days: {(merged['target'] == 0).sum():,} ({(merged['target'] == 0).mean()*100:.1f}%)\n")
            f.write(f"  - Average change: ${merged['next_day_change'].mean():.2f}\n")
            f.write(f"  - Std dev: ${merged['next_day_change'].std():.2f}\n\n")
        
        f.write("Feature Statistics:\n")
        numeric_cols = merged.select_dtypes(include=[float, int]).columns
        f.write(merged[numeric_cols].describe().to_string())
    
    print(f"[OK] Summary statistics saved to {stats_file}")
    
    # Final summary
    print_section("EDA COMPLETE", 60)
    print(f"\nGenerated Visualizations:")
    print(f"  [OK] results/basic_eda.png")
    print(f"  [OK] results/advanced_eda.png")
    print(f"  [OK] results/correlation_matrix.png")
    print(f"  [OK] results/sentiment_analysis.png")
    print(f"  [OK] results/data_summary_statistics.txt")
    print(f"\nAll results saved in: {RESULTS_DIR}")
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

