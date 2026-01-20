"""
Main script for Stock Sentiment Analysis
Run this script to perform complete analysis from data loading to model evaluation
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
from src.embeddings import (
    prepare_tokens, train_word2vec_on_train, create_w2v_embeddings,
    train_tfidf_on_train, create_tfidf_embeddings
)
from src.models import (
    prepare_features, train_test_split_data, train_all_models, 
    evaluate_model, apply_feature_selection
)
from src.visualization import (
    plot_basic_eda, plot_advanced_eda, plot_correlation_matrix,
    plot_sentiment_analysis, plot_model_evaluation, plot_feature_importance
)
from src.config import MERGED_OUTPUT, PREDICTIONS_OUTPUT
from src.utils import print_section
from tqdm import tqdm
import pandas as pd


def main():
    """Main execution function."""
    print_section("STOCK SENTIMENT ANALYSIS - STARTING", 60)
    
    # Overall progress tracking
    total_steps = 16
    main_progress = tqdm(total=total_steps, desc="Overall Progress", position=0, leave=True)
    
    # Step 1: Load and preprocess data
    print_section("STEP 1: DATA LOADING AND PREPROCESSING", 60)
    tweets, prices = load_data()
    tweets, prices = preprocess_data(tweets, prices)
    get_data_summary(tweets, prices)
    main_progress.update(1)
    
    # Step 2: Basic EDA
    print_section("STEP 2: BASIC EXPLORATORY DATA ANALYSIS", 60)
    plot_basic_eda(tweets, prices)
    main_progress.update(1)
    
    # Step 3: Merge datasets
    print_section("STEP 3: MERGING DATASETS", 60)
    merged = merge_datasets(tweets, prices)
    main_progress.update(1)
    
    # Step 4: Compute sentiment features
    print_section("STEP 4: SENTIMENT ANALYSIS", 60)
    merged = compute_all_sentiment_features(merged)
    main_progress.update(1)
    
    # Display tweets with sentiment labels
    from src.sentiment_display import display_tweets_with_sentiment, save_tweets_with_sentiment_csv
    print_section("STEP 4.5: DISPLAYING TWEETS WITH SENTIMENT LABELS", 60)
    display_tweets_with_sentiment(merged, n_samples=20, save_to_file=True)
    save_tweets_with_sentiment_csv(merged)
    main_progress.update(1)
    
    # Step 5: Create target variable
    print_section("STEP 5: CREATING TARGET VARIABLE", 60)
    merged = create_target_variable(merged)
    main_progress.update(1)
    
    # Step 6: Advanced EDA
    print_section("STEP 6: ADVANCED EXPLORATORY DATA ANALYSIS", 60)
    plot_advanced_eda(tweets, prices)
    plot_correlation_matrix(merged)
    plot_sentiment_analysis(merged)
    main_progress.update(1)
    
    # Step 7: Feature engineering - technical indicators
    print_section("STEP 7: FEATURE ENGINEERING", 60)
    merged = compute_technical_indicators(merged)
    main_progress.update(1)
    
    # Step 8: Prepare tokens for embeddings
    print_section("STEP 8: PREPARING TOKENS FOR EMBEDDINGS", 60)
    merged = prepare_tokens(merged)
    main_progress.update(1)
    
    # Step 9: Split data BEFORE training embeddings (prevent data leakage)
    # Use time series split to ensure test set is from later period (no lookahead bias)
    print_section("STEP 9: TIME SERIES DATA SPLITTING (BEFORE EMBEDDING TRAINING)", 60)
    print("⚠️  IMPORTANT: Splitting data BEFORE training embeddings to prevent data leakage")
    print("🔒 Using time series split: test set will be from later period (no lookahead bias)")
    
    from src.model_pipelines import create_time_series_split
    from src.config import TEST_SIZE
    
    # Use time series split (test set from later period)
    train_indices, test_indices = create_time_series_split(merged, test_size=TEST_SIZE)
    
    merged_train = merged.loc[train_indices].copy()
    merged_test = merged.loc[test_indices].copy()
    
    print(f"✅ Training set: {len(merged_train):,} samples")
    print(f"✅ Test set: {len(merged_test):,} samples")
    main_progress.update(1)
    
    # Step 10: Train embeddings ONLY on training data
    print_section("STEP 10: TRAINING EMBEDDINGS (TRAINING DATA ONLY)", 60)
    print("🔒 Training Word2Vec and TF-IDF ONLY on training data to prevent data leakage...")
    
    # Train Word2Vec on training data only
    train_tokens = merged_train['tokens']
    w2v_model = train_word2vec_on_train(train_tokens)
    
    # Train TF-IDF on training data only
    from src.config import TFIDF_MAX_FEATURES
    # Get tweet column name (handle both 'tweet' and 'Tweet')
    tweet_col = 'tweet' if 'tweet' in merged_train.columns else 'Tweet'
    if tweet_col not in merged_train.columns:
        raise KeyError(f"Tweet column not found. Available columns: {merged_train.columns.tolist()}")
    train_tweets = merged_train[tweet_col]
    tfidf_vectorizer = train_tfidf_on_train(train_tweets, max_features=TFIDF_MAX_FEATURES)
    main_progress.update(1)
    
    # Step 11: Create embeddings for both train and test using trained models
    print_section("STEP 11: CREATING EMBEDDINGS FOR TRAIN AND TEST", 60)
    
    # Word2Vec embeddings
    print("Creating Word2Vec embeddings for training set...")
    w2v_train = create_w2v_embeddings(merged_train['tokens'], w2v_model)
    print("Creating Word2Vec embeddings for test set...")
    w2v_test = create_w2v_embeddings(merged_test['tokens'], w2v_model)
    
    # TF-IDF embeddings
    print("Creating TF-IDF embeddings for training set...")
    tfidf_train = create_tfidf_embeddings(merged_train[tweet_col], tfidf_vectorizer)
    print("Creating TF-IDF embeddings for test set...")
    tfidf_test = create_tfidf_embeddings(merged_test[tweet_col], tfidf_vectorizer)
    main_progress.update(1)
    
    # Step 12: Prepare features for ML
    print_section("STEP 12: PREPARING FEATURES FOR MACHINE LEARNING", 60)
    
    # Prepare features for training set
    X_train, y_train = prepare_features(
        merged_train, 
        w2v_embeddings=w2v_train,
        tfidf_embeddings=tfidf_train
    )
    
    # Prepare features for test set
    X_test, y_test = prepare_features(
        merged_test,
        w2v_embeddings=w2v_test,
        tfidf_embeddings=tfidf_test
    )
    
    print(f"✅ Training features shape: {X_train.shape}")
    print(f"✅ Test features shape: {X_test.shape}")
    main_progress.update(1)
    
    # Step 12.5: Feature Selection parameters (will be used inside CV loop)
    print_section("STEP 12.5: FEATURE SELECTION CONFIGURATION", 60)
    from src.config import (
        USE_FEATURE_SELECTION, FEATURE_SELECTION_METHOD, 
        FEATURE_SELECTION_N_FEATURES, FEATURE_SELECTION_CV_FOLDS
    )
    
    feature_selection_params = None
    if USE_FEATURE_SELECTION:
        feature_selection_params = {
            'method': FEATURE_SELECTION_METHOD,
            'n_features': FEATURE_SELECTION_N_FEATURES,
            'cv_folds': FEATURE_SELECTION_CV_FOLDS
        }
        print(f"✅ Feature selection enabled: {FEATURE_SELECTION_METHOD}")
        print(f"   (Will be applied inside cross-validation loop)")
    else:
        print("⚠️  Feature selection disabled - using all features")
    
    # Step 13: Train models using pipelines with TimeSeriesSplit CV
    print_section("STEP 13: TRAINING MODELS WITH PIPELINES & TIME SERIES CV", 60)
    print("🔒 Using scikit-learn Pipeline and ColumnTransformer for safe preprocessing")
    print("🔒 Using TimeSeriesSplit for cross-validation (training always precedes test)")
    print("🔒 All preprocessing (scaling, feature selection) wrapped inside CV loop")
    
    from src.model_pipelines import train_all_models_pipelines
    from src.config import FEATURE_SELECTION_CV_FOLDS
    
    results, best_model_name = train_all_models_pipelines(
        X_train, X_test, y_train, y_test,
        use_feature_selection=USE_FEATURE_SELECTION,
        feature_selection_params=feature_selection_params,
        cv_folds=FEATURE_SELECTION_CV_FOLDS,
        use_time_series_cv=True  # Use TimeSeriesSplit instead of random splits
    )
    main_progress.update(1)
    
    # Step 14: Detailed evaluation of all models
    print_section("STEP 14: DETAILED MODEL EVALUATION", 60)
    from src.model_pipelines import evaluate_model_detailed
    
    evaluations = {}
    for model_name in results.keys():
        evaluations[model_name] = evaluate_model_detailed(
            results[model_name], model_name, y_test, X_test
        )
    
    # Step 15: Comprehensive visualizations
    print_section("STEP 15: COMPREHENSIVE MODEL EVALUATION VISUALIZATIONS", 60)
    
    # Model evaluation plots (all models)
    plot_model_evaluation(results, best_model_name, y_test)
    
    # Feature importance (best model)
    best_model = results[best_model_name]['model']
    best_pipeline = results[best_model_name]['pipeline']
    top_features = plot_feature_importance(best_model, X_train, best_model_name, pipeline=best_pipeline)
    
    if top_features is not None:
        print("\n📊 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features.head(10).items(), 1):
            print(f"  {i:2d}. {feature:30s} {importance:.4f}")
    
    # Error analysis
    from src.visualization import plot_error_analysis
    plot_error_analysis(results, best_model_name, y_test, merged_test)
    
    # Sentiment distributions
    from src.visualization import plot_sentiment_distributions
    plot_sentiment_distributions(
        merged_train, merged_test, 
        y_test, results[best_model_name]['predictions']
    )
    main_progress.update(1)
    
    # Step 16: Save results
    print_section("STEP 16: SAVING RESULTS", 60)
    
    # Combine train and test data for saving
    merged_train['w2v_embeddings'] = [list(row) for row in w2v_train.values]
    merged_test['w2v_embeddings'] = [list(row) for row in w2v_test.values]
    merged_all = pd.concat([merged_train, merged_test], ignore_index=True)
    
    # Ensure sentiment labels are included in saved data
    print("✅ Saving merged data with sentiment labels (tb_label, vader_label)...")
    merged_all.to_csv(MERGED_OUTPUT, index=False)
    print(f'✅ Saved merged data: {MERGED_OUTPUT}')
    print(f'   Includes sentiment labels: tb_label (TextBlob), vader_label (VADER)')
    
    # Save predictions for all models
    test_data = merged_test.copy()
    for model_name, model_results in results.items():
        test_data[f'predicted_{model_name.lower().replace(" ", "_")}'] = model_results['predictions']
        test_data[f'probability_{model_name.lower().replace(" ", "_")}'] = model_results['probabilities']
    
    # Also save best model predictions with simpler column names
    test_data['predicted'] = results[best_model_name]['predictions']
    test_data['probability'] = results[best_model_name]['probabilities']
    
    if 'date' in test_data.columns:
        # Save all predictions with sentiment labels
        pred_cols = ['date', 'stock name']
        
        # Add tweet column if available
        tweet_col = 'tweet' if 'tweet' in test_data.columns else 'Tweet'
        if tweet_col in test_data.columns:
            pred_cols.append(tweet_col)
        
        # Add sentiment labels
        if 'tb_label' in test_data.columns:
            pred_cols.append('tb_label')
        if 'vader_label' in test_data.columns:
            pred_cols.append('vader_label')
        
        # Add target and predictions
        pred_cols.extend(['target', 'predicted', 'probability'])
        pred_cols.extend([f'predicted_{m.lower().replace(" ", "_")}' for m in results.keys()])
        pred_cols.extend([f'probability_{m.lower().replace(" ", "_")}' for m in results.keys()])
        pred_cols = [c for c in pred_cols if c in test_data.columns]
        test_data[pred_cols].to_csv(PREDICTIONS_OUTPUT, index=False)
        print(f'✅ Saved predictions with sentiment labels: {PREDICTIONS_OUTPUT}')
        print(f'   Includes: tweets, sentiment labels (tb_label, vader_label), predictions')
    
    # Final summary
    print_section("FINAL SUMMARY", 60)
    print(f"\n📊 Dataset Summary:")
    print(f"  - Total merged records: {len(merged_all):,}")
    print(f"  - Features used: {X_train.shape[1]}")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Test samples: {len(X_test):,}")
    print(f"\n🔒 Data Leakage Prevention:")
    print(f"  ✅ Word2Vec trained ONLY on training data")
    print(f"  ✅ TF-IDF fitted ONLY on training data")
    print(f"  ✅ Test embeddings created using models trained on training data only")
    print(f"  ✅ Feature engineering uses only data available at prediction time")
    print(f"  ✅ Technical indicators are backward-looking (no future data)")
    print(f"  ✅ Time series split: test set from later period (no lookahead bias)")
    print(f"  ✅ TimeSeriesSplit used for cross-validation (training always precedes test)")
    print(f"  ✅ All preprocessing (scaling, feature selection) wrapped in Pipeline")
    print(f"  ✅ All transformations inside CV loop (no test data leakage)")
    if USE_FEATURE_SELECTION:
        print(f"  ✅ Feature selection performed inside cross-validation loop")
    
    print(f"\n🎯 Model Performance Summary:")
    print(f"{'Model':<25} {'CV Accuracy':<15} {'CV ROC-AUC':<15} {'Test Accuracy':<15} {'Test ROC-AUC':<15}")
    print("-" * 85)
    for model_name, metrics in results.items():
        marker = "🏆" if model_name == best_model_name else "  "
        cv_acc = metrics.get('cv_accuracy_mean', 0)
        cv_auc = metrics.get('cv_roc_auc_mean', 0)
        print(f"{marker} {model_name:<23} {cv_acc:<15.4f} {cv_auc:<15.4f} "
              f"{metrics['accuracy']:<15.4f} {metrics['roc_auc']:<15.4f}")
    
    main_progress.update(1)  # Final step
    main_progress.close()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE! 🎉")
    print("="*60)
    print(f"\nResults saved in: {MERGED_OUTPUT.parent}")


if __name__ == "__main__":
    main()

