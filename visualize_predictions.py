"""
Script to visualize predictions from a CSV file with predictions.
Usage: python visualize_predictions.py --data predictions.csv [--true-labels target]
"""
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.visualization import plot_predictions
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions from CSV file')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to CSV file with predictions (must have "predicted" and "prediction_probability" columns)')
    parser.add_argument('--true-labels', type=str, 
                       help='Column name for true labels (optional, for evaluation metrics)')
    parser.add_argument('--date-col', type=str, 
                       help='Column name for date (optional, defaults to "date" or "Date")')
    parser.add_argument('--stock-col', type=str, 
                       help='Column name for stock (optional, defaults to "stock name" or "Stock Name")')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from: {args.data}")
    df = pd.read_csv(args.data)
    
    # Check required columns
    if 'predicted' not in df.columns:
        raise ValueError("CSV file must have 'predicted' column")
    if 'prediction_probability' not in df.columns:
        raise ValueError("CSV file must have 'prediction_probability' column")
    
    # Extract true labels if provided
    true_labels = None
    if args.true_labels:
        if args.true_labels not in df.columns:
            print(f"⚠️  Warning: Column '{args.true_labels}' not found. Skipping evaluation metrics.")
        else:
            true_labels = df[args.true_labels].values
            print(f"📊 Using '{args.true_labels}' column as true labels for evaluation")
    elif 'target' in df.columns:
        true_labels = df['target'].values
        print("📊 Using 'target' column as true labels for evaluation")
    
    # Generate visualizations
    plot_predictions(df, true_labels=true_labels, 
                    date_col=args.date_col, stock_col=args.stock_col)
    
    print(f"\n✅ Visualizations saved to: {RESULTS_DIR / 'predictions_analysis.png'}")


if __name__ == "__main__":
    main()

