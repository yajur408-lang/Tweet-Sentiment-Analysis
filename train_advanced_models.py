"""
Advanced Model Training Script
Includes hyperparameter tuning, ensemble methods, SHAP explanations, and model saving
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_data, preprocess_data, merge_datasets, create_target_variable
from src.sentiment_analysis import compute_all_sentiment_features
from src.feature_engineering import compute_technical_indicators
from src.embeddings import (
    prepare_tokens, train_word2vec_on_train, create_w2v_embeddings,
    train_tfidf_on_train, create_tfidf_embeddings
)
from src.models import prepare_features
from src.model_pipelines import create_time_series_split, train_all_models_pipelines
from src.advanced_evaluation import (
    run_comprehensive_cv, hyperparameter_tuning, create_ensemble_voting,
    create_ensemble_stacking, evaluate_ensemble, create_model_comparison_table,
    analyze_misclassifications, plot_comprehensive_evaluation,
    generate_shap_explanations, plot_shap_summary
)
from src.visualization import plot_error_analysis, plot_sentiment_distributions
from src.config import RESULTS_DIR, RANDOM_STATE, TEST_SIZE
from src.utils import print_section


def main():
    """Main execution function for advanced model training."""
    print_section("ADVANCED MODEL TRAINING - STARTING", 60)
    
    # Step 1: Load and prepare data (same as main.py)
    print_section("STEP 1: DATA LOADING AND PREPARATION", 60)
    tweets, prices = load_data()
    tweets, prices = preprocess_data(tweets, prices)
    merged = merge_datasets(tweets, prices)
    merged = compute_all_sentiment_features(merged)
    merged = create_target_variable(merged)
    merged = compute_technical_indicators(merged)
    merged = prepare_tokens(merged)
    
    # Step 2: Time series split
    print_section("STEP 2: TIME SERIES DATA SPLITTING", 60)
    train_indices, test_indices = create_time_series_split(merged, test_size=TEST_SIZE)
    merged_train = merged.loc[train_indices].copy()
    merged_test = merged.loc[test_indices].copy()
    
    # Step 3: Train embeddings
    print_section("STEP 3: TRAINING EMBEDDINGS", 60)
    from src.config import TFIDF_MAX_FEATURES
    
    # Get tweet column name (handle both 'tweet' and 'Tweet')
    tweet_col = 'tweet' if 'tweet' in merged_train.columns else 'Tweet'
    if tweet_col not in merged_train.columns:
        raise KeyError(f"Tweet column not found. Available columns: {merged_train.columns.tolist()}")
    
    w2v_model = train_word2vec_on_train(merged_train['tokens'])
    tfidf_vectorizer = train_tfidf_on_train(merged_train[tweet_col], max_features=TFIDF_MAX_FEATURES)
    
    # Save embedding models for prediction
    joblib.dump(w2v_model, RESULTS_DIR / 'w2v_model.pkl')
    joblib.dump(tfidf_vectorizer, RESULTS_DIR / 'tfidf_vectorizer.pkl')
    print(f"✅ Saved embedding models for prediction")
    
    w2v_train = create_w2v_embeddings(merged_train['tokens'], w2v_model)
    w2v_test = create_w2v_embeddings(merged_test['tokens'], w2v_model)
    tfidf_train = create_tfidf_embeddings(merged_train[tweet_col], tfidf_vectorizer)
    tfidf_test = create_tfidf_embeddings(merged_test[tweet_col], tfidf_vectorizer)
    
    # Step 4: Prepare features
    print_section("STEP 4: PREPARING FEATURES", 60)
    X_train, y_train = prepare_features(merged_train, w2v_embeddings=w2v_train, tfidf_embeddings=tfidf_train)
    X_test, y_test = prepare_features(merged_test, w2v_embeddings=w2v_test, tfidf_embeddings=tfidf_test)
    
    print(f"✅ Training features: {X_train.shape}")
    print(f"✅ Test features: {X_test.shape}")
    
    # Step 5: Train baseline models
    print_section("STEP 5: TRAINING BASELINE MODELS", 60)
    from src.config import USE_FEATURE_SELECTION, FEATURE_SELECTION_METHOD, FEATURE_SELECTION_N_FEATURES, FEATURE_SELECTION_CV_FOLDS
    
    feature_selection_params = None
    if USE_FEATURE_SELECTION:
        feature_selection_params = {
            'method': FEATURE_SELECTION_METHOD,
            'n_features': FEATURE_SELECTION_N_FEATURES,
            'cv_folds': FEATURE_SELECTION_CV_FOLDS
        }
    
    baseline_results, best_baseline_name = train_all_models_pipelines(
        X_train, X_test, y_train, y_test,
        use_feature_selection=USE_FEATURE_SELECTION,
        feature_selection_params=feature_selection_params,
        cv_folds=5,
        use_time_series_cv=True
    )
    
    # Step 6: Comprehensive CV evaluation
    print_section("STEP 6: COMPREHENSIVE CROSS-VALIDATION", 60)
    cv_results_all = {}
    for model_name, results in baseline_results.items():
        pipeline = results['pipeline']
        cv_summary, cv_full = run_comprehensive_cv(
            pipeline, X_train, y_train, cv_folds=5
        )
        cv_results_all[model_name] = cv_summary
        baseline_results[model_name].update({
            'cv_summary': cv_summary,
            'cv_roc_auc_mean': cv_summary['roc_auc']['test_mean'],
            'cv_roc_auc_std': cv_summary['roc_auc']['test_std']
        })
        
        print(f"\n{model_name} CV Results:")
        for metric, stats in cv_summary.items():
            print(f"  {metric}: {stats['test_mean']:.4f} (+/- {stats['test_std']*2:.4f})")
    
    # Step 7: Hyperparameter tuning for top 2 models
    print_section("STEP 7: HYPERPARAMETER TUNING", 60)
    
    # Get top 2 models by ROC-AUC
    sorted_models = sorted(baseline_results.items(), 
                          key=lambda x: x[1]['roc_auc'], 
                          reverse=True)
    top_2_models = sorted_models[:2]
    
    tuned_results = {}
    for model_name, baseline_result in top_2_models:
        print(f"\n{'='*60}")
        print(f"TUNING {model_name}")
        print(f"{'='*60}")
        
        pipeline = baseline_result['pipeline']
        
        # Define parameter grids
        if 'Logistic Regression' in model_name:
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0, 100.0],
                'classifier__max_iter': [500, 1000, 2000]
            }
        elif 'Random Forest' in model_name:
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        elif 'XGBoost' in model_name:
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            print(f"⚠️  No parameter grid defined for {model_name}, skipping tuning")
            continue
        
        # Tune (use RandomizedSearchCV for speed)
        best_pipeline, best_params, best_cv_score = hyperparameter_tuning(
            pipeline, param_grid, X_train, y_train,
            cv_folds=5, method='random', n_iter=20
        )
        
        # Evaluate tuned model on test set
        best_pipeline.fit(X_train, y_train)
        y_pred_tuned = best_pipeline.predict(X_test)
        y_proba_tuned = best_pipeline.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        tuned_results[f"{model_name} (Tuned)"] = {
            'pipeline': best_pipeline,
            'model': best_pipeline.named_steps['classifier'],
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'accuracy': accuracy_score(y_test, y_pred_tuned),
            'roc_auc': roc_auc_score(y_test, y_proba_tuned),
            'precision': precision_score(y_test, y_pred_tuned),
            'recall': recall_score(y_test, y_pred_tuned),
            'f1_score': f1_score(y_test, y_pred_tuned),
            'predictions': y_pred_tuned,
            'probabilities': y_proba_tuned
        }
        
        print(f"\n✅ Tuned {model_name}:")
        print(f"   Baseline ROC-AUC: {baseline_result['roc_auc']:.4f}")
        print(f"   Tuned ROC-AUC:    {tuned_results[f'{model_name} (Tuned)']['roc_auc']:.4f}")
        print(f"   Improvement:      {tuned_results[f'{model_name} (Tuned)']['roc_auc'] - baseline_result['roc_auc']:.4f}")
    
    # Step 8: Ensemble methods
    print_section("STEP 8: ENSEMBLE METHODS", 60)
    
    # Get top models for ensemble
    all_results = {**baseline_results, **tuned_results}
    sorted_all = sorted(all_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    top_3_for_ensemble = sorted_all[:3]
    
    ensemble_results = {}
    
    # Voting Classifier
    print("\n" + "="*60)
    print("CREATING VOTING CLASSIFIER")
    print("="*60)
    voting_pipelines = {name: result['pipeline'] for name, result in top_3_for_ensemble}
    voting_ensemble = create_ensemble_voting(voting_pipelines, voting='soft')
    voting_result = evaluate_ensemble(voting_ensemble, X_train, y_train, X_test, y_test, "Voting Classifier")
    ensemble_results['Voting Classifier'] = voting_result
    
    # Stacking Classifier
    print("\n" + "="*60)
    print("CREATING STACKING CLASSIFIER")
    print("="*60)
    stacking_estimators = [(name, result['pipeline']) for name, result in top_3_for_ensemble]
    stacking_ensemble = create_ensemble_stacking(stacking_estimators)
    stacking_result = evaluate_ensemble(stacking_ensemble, X_train, y_train, X_test, y_test, "Stacking Classifier")
    ensemble_results['Stacking Classifier'] = stacking_result
    
    # Step 9: Model comparison
    print_section("STEP 9: MODEL COMPARISON", 60)
    all_final_results = {**baseline_results, **tuned_results, **ensemble_results}
    comparison_df = create_model_comparison_table(all_final_results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\n✅ Saved model comparison to: {RESULTS_DIR / 'model_comparison.csv'}")
    
    # Step 10: Select best model
    print_section("STEP 10: SELECTING BEST MODEL", 60)
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_result = all_final_results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test ROC-AUC: {best_model_result['roc_auc']:.4f}")
    print(f"   Test Accuracy: {best_model_result['accuracy']:.4f}")
    
    # Step 11: Comprehensive evaluation plots
    print_section("STEP 11: CREATING EVALUATION PLOTS", 60)
    plot_comprehensive_evaluation(all_final_results, y_test)
    
    # Plot sentiment and target distributions (train vs test)
    plot_sentiment_distributions(merged_train, merged_test, y_test, best_model_result['predictions'])
    
    # Step 12: Error analysis
    print_section("STEP 12: ERROR ANALYSIS", 60)
    plot_error_analysis(all_final_results, best_model_name, y_test, merged_test)
    
    # Analyze misclassifications
    misclassified_df = analyze_misclassifications(
        y_test, best_model_result['predictions'], X_test, merged_test, n_samples=20
    )
    print(f"\n📊 Sample Misclassifications:")
    if len(misclassified_df) > 0:
        print(misclassified_df.head(10).to_string())
        misclassified_df.to_csv(RESULTS_DIR / 'misclassified_examples.csv', index=False)
        print(f"✅ Saved misclassified examples to: {RESULTS_DIR / 'misclassified_examples.csv'}")
    
    # Advanced error pattern analysis
    from src.error_pattern_analysis import analyze_error_patterns, document_error_findings
    error_analysis = analyze_error_patterns(
        y_test, best_model_result['predictions'], merged_test
    )
    findings = document_error_findings(error_analysis)
    
    # Step 13: SHAP explanations
    print_section("STEP 13: SHAP EXPLANATIONS", 60)
    best_model = best_model_result['model']
    
    # For ensemble models, use the first base estimator
    if hasattr(best_model, 'estimators_'):
        # It's an ensemble, use first estimator
        explainer_model = best_model.estimators_[0] if hasattr(best_model, 'estimators_') else best_model
    else:
        explainer_model = best_model
    
    shap_values, explainer = generate_shap_explanations(explainer_model, X_test, n_samples=100)
    if shap_values is not None:
        plot_shap_summary(shap_values, X_test.head(100))
        print("✅ SHAP explanations generated")
    else:
        print("⚠️  SHAP explanations not available")
    
    # Step 14: Save best model
    print_section("STEP 14: SAVING BEST MODEL", 60)
    model_save_path = RESULTS_DIR / 'best_model.pkl'
    pipeline_save_path = RESULTS_DIR / 'best_pipeline.pkl'
    
    # Save pipeline (includes all preprocessing)
    joblib.dump(best_model_result['pipeline'], pipeline_save_path)
    print(f"✅ Saved best pipeline to: {pipeline_save_path}")
    
    # Save model only (for SHAP if needed)
    if 'model' in best_model_result:
        joblib.dump(best_model_result['model'], model_save_path)
        print(f"✅ Saved best model to: {model_save_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'test_roc_auc': float(best_model_result['roc_auc']),
        'test_accuracy': float(best_model_result['accuracy']),
        'test_precision': float(best_model_result.get('precision', 0)),
        'test_recall': float(best_model_result.get('recall', 0)),
        'test_f1_score': float(best_model_result.get('f1_score', 0)),
        'feature_count': X_train.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import json
    with open(RESULTS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved model metadata to: {RESULTS_DIR / 'model_metadata.json'}")
    
    # Step 15: Prediction visualizations
    print_section("STEP 15: CREATING PREDICTION VISUALIZATIONS", 60)
    from src.visualization import plot_predictions
    
    # Create predictions dataframe
    predictions_df = merged_test.copy()
    predictions_df['predicted'] = best_model_result['predictions']
    predictions_df['prediction_probability'] = best_model_result['probabilities']
    
    # Generate comprehensive prediction plots
    plot_predictions(predictions_df, true_labels=y_test)
    
    # Save predictions with all metadata
    pred_output_path = RESULTS_DIR / 'test_predictions.csv'
    pred_cols = ['predicted', 'prediction_probability', 'target']
    if 'date' in predictions_df.columns:
        pred_cols.insert(0, 'date')
    if 'stock name' in predictions_df.columns:
        pred_cols.insert(1, 'stock name')
    elif 'Stock Name' in predictions_df.columns:
        pred_cols.insert(1, 'Stock Name')
    if 'tweet' in predictions_df.columns:
        pred_cols.append('tweet')
    elif 'Tweet' in predictions_df.columns:
        pred_cols.append('Tweet')
    
    available_cols = [c for c in pred_cols if c in predictions_df.columns]
    predictions_df[available_cols].to_csv(pred_output_path, index=False)
    print(f"✅ Saved test predictions to: {pred_output_path}")
    
    # Step 16: Final summary
    print_section("FINAL SUMMARY", 60)
    print(f"\n📊 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {best_model_result['roc_auc']:.4f}")
    print(f"   Accuracy: {best_model_result['accuracy']:.4f}")
    print(f"   Precision: {best_model_result.get('precision', 0):.4f}")
    print(f"   Recall: {best_model_result.get('recall', 0):.4f}")
    print(f"   F1-Score: {best_model_result.get('f1_score', 0):.4f}")
    
    print(f"\n📁 Saved Files:")
    print(f"   - Best pipeline: {pipeline_save_path}")
    print(f"   - Model comparison: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"   - Test predictions: {RESULTS_DIR / 'test_predictions.csv'}")
    print(f"   - Prediction visualizations: {RESULTS_DIR / 'predictions_analysis.png'}")
    print(f"   - Misclassified examples: {RESULTS_DIR / 'misclassified_examples.csv'}")
    print(f"   - Model metadata: {RESULTS_DIR / 'model_metadata.json'}")
    
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING COMPLETE! 🎉")
    print("="*60)
    
    return best_model_result, all_final_results


if __name__ == "__main__":
    best_model, all_results = main()

