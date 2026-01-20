"""
Prediction function using saved best model
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import RESULTS_DIR
from src.embeddings import create_w2v_embeddings, create_tfidf_embeddings
from src.models import prepare_features
from src.sentiment_analysis import compute_all_sentiment_features
from src.feature_engineering import compute_technical_indicators
from src.data_loader import merge_datasets, create_target_variable
from src.embeddings import prepare_tokens


def load_model(model_path=None):
    """
    Load the saved best model pipeline.
    
    Args:
        model_path: Path to saved model (default: results/best_pipeline.pkl)
    
    Returns:
        Loaded pipeline
    """
    if model_path is None:
        model_path = RESULTS_DIR / 'best_pipeline.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    pipeline = joblib.load(model_path)
    print(f"✅ Loaded model from: {model_path}")
    return pipeline


def predict_single_tweet(tweet_text, stock_name, date, pipeline, 
                         w2v_model=None, tfidf_vectorizer=None, 
                         price_data=None):
    """
    Predict price direction for a single tweet.
    
    Args:
        tweet_text: Tweet text
        stock_name: Stock name
        date: Date of tweet
        pipeline: Trained pipeline
        w2v_model: Word2Vec model (optional, will load if None)
        tfidf_vectorizer: TF-IDF vectorizer (optional, will load if None)
        price_data: Price data for technical indicators (optional)
    
    Returns:
        dict: Prediction results
    """
    # This is a simplified version - in production, you'd need to:
    # 1. Compute sentiment features
    # 2. Create embeddings
    # 3. Compute technical indicators
    # 4. Prepare feature vector
    # 5. Predict
    
    print("⚠️  Single tweet prediction requires full feature engineering pipeline.")
    print("   For now, use predict_batch() with a dataframe.")
    
    return None


def predict_batch(df, pipeline, w2v_model=None, tfidf_vectorizer=None):
    """
    Predict for a batch of tweets (dataframe with same structure as training data).
    
    Args:
        df: DataFrame with tweets and required features
        pipeline: Trained pipeline
        w2v_model: Word2Vec model (optional)
        tfidf_vectorizer: TF-IDF vectorizer (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Load models if not provided
    if w2v_model is None:
        w2v_model_path = RESULTS_DIR / 'w2v_model.pkl'
        if w2v_model_path.exists():
            w2v_model = joblib.load(w2v_model_path)
    
    if tfidf_vectorizer is None:
        tfidf_model_path = RESULTS_DIR / 'tfidf_vectorizer.pkl'
        if tfidf_model_path.exists():
            tfidf_vectorizer = joblib.load(tfidf_model_path)
    
    # Prepare features (same as training)
    # Note: This assumes df has all required features already computed
    # In practice, you'd need to run the full preprocessing pipeline
    
    print("⚠️  Batch prediction requires full feature engineering.")
    print("   The dataframe must have all features computed (sentiment, embeddings, technical indicators).")
    
    return None


def predict_from_merged_data(merged_data, pipeline):
    """
    Predict using merged data (with all features already computed).
    
    Args:
        merged_data: DataFrame with all features (same as training data structure)
        pipeline: Trained pipeline
    
    Returns:
        DataFrame with predictions added
    """
    from src.models import prepare_features
    
    # Prepare features
    X, y = prepare_features(merged_data)
    
    # Predict
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]
    
    # Add to dataframe
    result_df = merged_data.copy()
    result_df['predicted'] = predictions
    result_df['prediction_probability'] = probabilities
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using saved model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--data', type=str, help='Path to data CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--plot', action='store_true', help='Generate prediction visualizations')
    parser.add_argument('--true-labels', type=str, help='Column name for true labels (optional, for evaluation)')
    
    args = parser.parse_args()
    
    # Load model
    pipeline = load_model(args.model)
    
    if args.data:
        # Load data and predict
        df = pd.read_csv(args.data)
        result_df = predict_from_merged_data(df, pipeline)
        
        # Save results
        output_path = args.output or (RESULTS_DIR / 'predictions.csv')
        result_df.to_csv(output_path, index=False)
        print(f"✅ Saved predictions to: {output_path}")
        
        # Generate visualizations if requested
        if args.plot:
            from src.visualization import plot_predictions
            
            # Extract true labels if provided
            true_labels = None
            if args.true_labels and args.true_labels in result_df.columns:
                true_labels = result_df[args.true_labels].values
            elif 'target' in result_df.columns:
                true_labels = result_df['target'].values
                print("📊 Using 'target' column as true labels for evaluation")
            
            # Generate plots
            plot_predictions(result_df, true_labels=true_labels)
    else:
        print("Usage: python predict.py --data <path_to_data.csv> [--model <path_to_model.pkl>] [--output <output.csv>] [--plot] [--true-labels <column_name>]")

