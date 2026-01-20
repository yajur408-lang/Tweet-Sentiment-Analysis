"""
Example usage of individual modules
This script demonstrates how to use the modules separately
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_data, preprocess_data, merge_datasets
from src.sentiment_analysis import compute_all_sentiment_features
from src.feature_engineering import compute_technical_indicators
from src.models import prepare_features, train_test_split_data, train_xgboost

# Example: Load and preprocess data
print("Loading data...")
tweets, prices = load_data()
tweets, prices = preprocess_data(tweets, prices)

# Example: Merge datasets
print("\nMerging datasets...")
merged = merge_datasets(tweets, prices)

# Example: Compute sentiment
print("\nComputing sentiment...")
merged = compute_all_sentiment_features(merged)

# Example: Add technical indicators
print("\nComputing technical indicators...")
merged = compute_technical_indicators(merged)

print("\n✅ Example complete! Check merged dataframe:")
print(merged.head())



