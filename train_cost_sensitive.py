"""
Comprehensive cost-sensitive sentiment analysis training script.

This script demonstrates:
1. Cost matrix definition and visualization
2. Training multiple classical ML models with cost-aware evaluation
3. Class weights experimentation vs custom scorer
4. Threshold tuning for cost-sensitive decisions
5. Visualization of expensive mistakes
6. Hyperparameter tuning with cost-aware scoring
7. Deep learning models with custom loss
8. Model comparison and ensemble methods
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data, preprocess_data, merge_datasets, create_target_variable
from src.sentiment_analysis import compute_all_sentiment_features
from src.feature_engineering import compute_technical_indicators
from src.embeddings import (
    prepare_tokens, train_word2vec_on_train, create_w2v_embeddings,
    train_tfidf_on_train, create_tfidf_embeddings
)
from src.models import prepare_features, train_test_split_data
from src.cost_sensitive_evaluation import (
    COST_MATRIX, visualize_cost_matrix, compute_cost_score, compute_expected_cost,
    find_optimal_threshold, analyze_expensive_mistakes, print_cost_analysis,
    create_cost_aware_confusion_matrix_plot
)
from src.cost_aware_models import (
    train_all_cost_aware_models, experiment_class_weights,
    train_logistic_regression_weighted, train_linear_svm, train_naive_bayes,
    train_random_forest_cost_aware, train_xgboost_cost_aware,
    tune_hyperparameters_cost_aware
)
from src.deep_learning_models import (
    train_feedforward_nn_on_embeddings, create_model_comparison_table
)
from src.config import RESULTS_DIR, RANDOM_STATE
from src.utils import print_section

import matplotlib.pyplot as plt

# Print cost matrix
print_section("COST MATRIX DEFINITION", 80)
print("\nCost Matrix for Sentiment Classification:")
print("Row = True Class, Column = Predicted Class")
print(f"\n{COST_MATRIX}")
print("\nInterpretation:")
print("  - negative <-> positive = very bad (cost 5)")
print("  - negative <-> neutral or neutral <-> positive = less bad (cost 1)")
print("  - same class = no cost (0)")


def main():
    """Main training pipeline with cost-sensitive evaluation."""
    
    # Step 1: Load and preprocess data
    print_section("STEP 1: LOADING DATA", 80)
    tweets, prices = load_data()
    tweets, prices = preprocess_data(tweets, prices)
    merged = merge_datasets(tweets, prices)
    merged = create_target_variable(merged)
    
    # Step 2: Compute sentiment features
    print_section("STEP 2: COMPUTING SENTIMENT FEATURES", 80)
    merged = compute_all_sentiment_features(merged)
    
    # Create sentiment labels for classification (3-class: negative, neutral, positive)
    # Use VADER labels as ground truth sentiment
    from src.utils import vader_label
    if 'vader_label' not in merged.columns:
        merged['vader_label'] = merged['vader_compound'].apply(vader_label)
    
    print(f"\nSentiment Label Distribution:")
    print(merged['vader_label'].value_counts())
    
    # Step 3: Compute technical indicators
    print_section("STEP 3: COMPUTING TECHNICAL INDICATORS", 80)
    merged = compute_technical_indicators(merged)
    
    # Create sentiment labels for classification (3-class: negative, neutral, positive)
    # Use VADER labels as ground truth sentiment
    from src.utils import vader_label
    if 'vader_label' not in merged.columns:
        merged['vader_label'] = merged['vader_compound'].apply(vader_label)
    
    print(f"\nSentiment Label Distribution:")
    print(merged['vader_label'].value_counts())
    
    # Step 4: Prepare tokens for Word2Vec (if needed)
    print_section("STEP 4: PREPARING TOKENS", 80)
    merged = prepare_tokens(merged)
    
    # Step 5: Split data BEFORE training embeddings (prevent data leakage)
    print_section("STEP 5: SPLITTING DATA (TRAIN/TEST)", 80)
    sentiment_target = merged['vader_label'].copy()
    
    # First prepare basic features to get indices
    X_basic, _ = prepare_features(merged)
    
    # Align sentiment target with feature indices
    y = sentiment_target[X_basic.index]
    X_basic = X_basic.loc[y.index]
    
    # Split data
    X_train_basic, X_test_basic, y_train, y_test = train_test_split_data(X_basic, y)
    
    # Get train/test indices from merged dataframe
    train_indices = X_train_basic.index
    test_indices = X_test_basic.index
    merged_train = merged.loc[train_indices].copy()
    merged_test = merged.loc[test_indices].copy()
    
    # Step 6: Train embeddings on training data only
    print_section("STEP 6: TRAINING EMBEDDINGS (TRAINING DATA ONLY)", 80)
    tweet_col = 'tweet' if 'tweet' in merged_train.columns else 'Tweet'
    
    # Train TF-IDF on training data only
    from src.config import TFIDF_MAX_FEATURES
    tfidf_vectorizer = train_tfidf_on_train(merged_train[tweet_col], max_features=TFIDF_MAX_FEATURES)
    
    # Train Word2Vec on training data only (optional, can be slow)
    w2v_model = None
    try:
        if 'tokens' in merged_train.columns:
            w2v_model = train_word2vec_on_train(merged_train['tokens'])
        else:
            print("⚠️  No 'tokens' column found, skipping Word2Vec")
    except Exception as e:
        print(f"⚠️  Word2Vec training failed: {e}")
    
    # Step 7: Create embeddings for both train and test
    print_section("STEP 7: CREATING EMBEDDINGS FOR TRAIN AND TEST", 80)
    
    # TF-IDF embeddings
    tfidf_train = create_tfidf_embeddings(merged_train[tweet_col], tfidf_vectorizer)
    tfidf_test = create_tfidf_embeddings(merged_test[tweet_col], tfidf_vectorizer)
    
    # Word2Vec embeddings (if model was trained)
    w2v_train = None
    w2v_test = None
    if w2v_model is not None and 'tokens' in merged_train.columns:
        w2v_train = create_w2v_embeddings(merged_train['tokens'], w2v_model)
        w2v_test = create_w2v_embeddings(merged_test['tokens'], w2v_model)
    
    # Step 8: Prepare full feature matrices
    print_section("STEP 8: PREPARING FULL FEATURE MATRICES", 80)
    
    # Prepare features for training set
    # Note: prepare_features filters NaN values, so we need to get the sentiment labels
    # after filtering to ensure alignment
    if w2v_train is not None:
        X_train, _ = prepare_features(merged_train, w2v_embeddings=w2v_train, 
                                      tfidf_embeddings=tfidf_train)
    else:
        X_train, _ = prepare_features(merged_train, tfidf_embeddings=tfidf_train)
    
    # Prepare features for test set
    if w2v_test is not None:
        X_test, _ = prepare_features(merged_test, w2v_embeddings=w2v_test, 
                                     tfidf_embeddings=tfidf_test)
    else:
        X_test, _ = prepare_features(merged_test, tfidf_embeddings=tfidf_test)
    
    # Extract sentiment labels using the same indices as X_train and X_test
    # (prepare_features may filter out some rows with NaN values)
    y_train = merged_train.loc[X_train.index, 'vader_label'].reset_index(drop=True)
    y_test = merged_test.loc[X_test.index, 'vader_label'].reset_index(drop=True)
    
    # Reset X indices to match
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Sentiment distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Sentiment distribution (test): {y_test.value_counts().to_dict()}")
    
    # Step 9: Visualize cost matrix
    print_section("STEP 9: VISUALIZING COST MATRIX", 80)
    cost_matrix_path = RESULTS_DIR / 'cost_matrix.png'
    visualize_cost_matrix(save_path=str(cost_matrix_path))
    plt.close()
    
    # Step 10: Train classical ML models with cost-aware evaluation
    print_section("STEP 10: TRAINING CLASSICAL ML MODELS (COST-AWARE)", 80)
    classical_results = train_all_cost_aware_models(X_train, y_train, X_test, y_test)
    
    # Step 11: Experiment with class weights vs custom scorer
    print_section("STEP 11: CLASS WEIGHTS EXPERIMENT", 80)
    class_weight_results = experiment_class_weights(X_train, y_train, X_test, y_test)
    
    # Step 12: Threshold tuning
    print_section("STEP 12: THRESHOLD TUNING", 80)
    
    # Use best classical model for threshold tuning
    best_classical_model_name = min(classical_results.keys(), 
                                    key=lambda x: classical_results[x]['cost'])
    best_classical_model = classical_results[best_classical_model_name]
    
    print(f"\nTuning thresholds for: {best_classical_model_name}")
    threshold_results = find_optimal_threshold(
        y_test, best_classical_model['probabilities']
    )
    
    print(f"\nOptimal Threshold: {threshold_results['optimal_threshold']:.2f}")
    print(f"Optimal Cost: {threshold_results['optimal_cost']:.2f}")
    print(f"Original Cost (without threshold tuning): {best_classical_model['cost']:.2f}")
    
    # Step 13: Analyze expensive mistakes
    print_section("STEP 13: ANALYZING EXPENSIVE MISTAKES", 80)
    
    # Use best model predictions
    y_pred_best = best_classical_model['predictions']
    tweet_col_test = merged.loc[y_test.index, tweet_col] if tweet_col in merged.columns else None
    
    mistake_analysis = analyze_expensive_mistakes(
        y_test, y_pred_best, 
        texts=tweet_col_test.tolist() if tweet_col_test is not None else None,
        n_examples=20
    )
    
    print_cost_analysis(mistake_analysis)
    
    # Create confusion matrix plot
    confusion_matrix_path = RESULTS_DIR / 'cost_aware_confusion_matrix.png'
    create_cost_aware_confusion_matrix_plot(
        mistake_analysis, save_path=str(confusion_matrix_path)
    )
    plt.close()
    
    # Step 14: Hyperparameter tuning with cost-aware scoring
    print_section("STEP 14: HYPERPARAMETER TUNING (COST-AWARE)", 80)
    
    # Example: Tune Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    rf_pipeline = Pipeline([
        ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 15, 20, None],
        'rf__min_samples_split': [2, 5, 10]
    }
    
    print("\nTuning Random Forest hyperparameters (cost-aware)...")
    tuned_rf, best_params = tune_hyperparameters_cost_aware(
        rf_pipeline, param_grid, X_train, y_train,
        cv_folds=3, n_iter=20, use_random_search=True
    )
    
    # Evaluate tuned model
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['negative', 'neutral', 'positive'])
    y_test_encoded = le.transform(y_test)
    y_pred_tuned = tuned_rf.predict(X_test)
    y_proba_tuned = tuned_rf.predict_proba(X_test)
    
    tuned_cost = compute_cost_score(y_test_encoded, y_pred_tuned)
    print(f"\nTuned Random Forest Cost: {tuned_cost:.2f}")
    
    # Step 15: Deep Learning Models (if TensorFlow available)
    print_section("STEP 15: DEEP LEARNING MODELS", 80)
    
    deep_learning_results = {}
    
    # Prepare embeddings for deep learning
    # For feed-forward NN, we'll use averaged Word2Vec embeddings if available
    if w2v_train is not None:
        # Use averaged Word2Vec embeddings
        X_train_w2v = w2v_train.values
        X_test_w2v = w2v_test.values
        
        # Average embeddings if needed (if they're sequences)
        if len(X_train_w2v.shape) > 2:
            X_train_w2v = np.mean(X_train_w2v, axis=1)
            X_test_w2v = np.mean(X_test_w2v, axis=1)
        
        try:
            ff_nn_result = train_feedforward_nn_on_embeddings(
                X_train_w2v, y_train, X_test_w2v, y_test,
                embedding_method='mean',
                hidden_dims=[128, 64],
                dropout_rate=0.5,
                epochs=20,
                batch_size=32,
                use_custom_loss=True
            )
            
            if ff_nn_result is not None:
                deep_learning_results['Feed-forward NN'] = ff_nn_result
                print(f"\nFeed-forward NN Cost: {ff_nn_result['cost']:.2f}")
        except Exception as e:
            print(f"⚠️  Feed-forward NN training failed: {e}")
    
    # Step 16: Model comparison
    print_section("STEP 16: MODEL COMPARISON", 80)
    
    # Combine all results
    all_results = {**classical_results, **deep_learning_results}
    
    # Add tuned Random Forest
    all_results['Random Forest (Tuned)'] = {
        'model': tuned_rf,
        'predictions': y_pred_tuned,
        'probabilities': y_proba_tuned,
        'accuracy': best_classical_model['accuracy'],  # Approximate
        'cost': tuned_cost,
        'expected_cost': compute_expected_cost(y_test_encoded, y_proba_tuned),
        'f1_macro': best_classical_model['f1_macro'],  # Approximate
        'confusion_matrix': best_classical_model['confusion_matrix']
    }
    
    # Create comparison table
    comparison_df = create_model_comparison_table(all_results)
    
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON (Sorted by Cost)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_csv_path = RESULTS_DIR / 'cost_sensitive_model_comparison.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n[OK] Saved comparison table to: {comparison_csv_path}")
    
    # Step 17: Summary
    print_section("STEP 17: SUMMARY", 80)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_cost = comparison_df.iloc[0]['Cost']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 Best Model (Lowest Cost): {best_model_name}")
    print(f"   Cost: {best_cost:.2f}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Expected Cost: {all_results[best_model_name]['expected_cost']:.4f}")
    
    print(f"\n📊 Results saved to: {RESULTS_DIR}")
    print(f"   - Cost matrix: cost_matrix.png")
    print(f"   - Confusion matrix: cost_aware_confusion_matrix.png")
    print(f"   - Model comparison: cost_sensitive_model_comparison.csv")
    
    return all_results, comparison_df


if __name__ == '__main__':
    results, comparison_df = main()
    print("\n[OK] Cost-sensitive training complete!")

