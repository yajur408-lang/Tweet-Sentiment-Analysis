"""
Advanced model evaluation with hyperparameter tuning, ensemble methods, and SHAP explanations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    cross_validate
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from .config import RANDOM_STATE, RESULTS_DIR


def run_comprehensive_cv(pipeline, X_train, y_train, cv_folds=5, scoring=None):
    """
    Run comprehensive 5-fold TimeSeriesSplit cross-validation.
    
    Args:
        pipeline: sklearn Pipeline
        X_train: Training features
        y_train: Training target
        cv_folds: Number of CV folds
        scoring: List of scoring metrics
    
    Returns:
        dict: CV results with mean and std
    """
    if scoring is None:
        scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    
    cv = TimeSeriesSplit(n_splits=cv_folds)
    
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results_summary = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        results_summary[metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std()
        }
    
    return results_summary, cv_results


def hyperparameter_tuning(pipeline, param_grid, X_train, y_train, 
                         cv_folds=5, method='grid', n_iter=50):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    
    Args:
        pipeline: sklearn Pipeline
        param_grid: Parameter grid for tuning
        X_train: Training features
        y_train: Training target
        cv_folds: Number of CV folds
        method: 'grid' or 'random'
        n_iter: Number of iterations for RandomizedSearchCV
    
    Returns:
        Best pipeline and results
    """
    cv = TimeSeriesSplit(n_splits=cv_folds)
    
    if method == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
    else:  # random
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_iter=n_iter,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE
        )
    
    print(f"Running {method} search with {cv_folds}-fold TimeSeriesSplit...")
    search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {search.best_params_}")
    print(f"✅ Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def create_ensemble_voting(pipelines_dict, voting='soft'):
    """
    Create Voting Classifier ensemble.
    
    Args:
        pipelines_dict: Dictionary of {name: pipeline}
        voting: 'soft' or 'hard'
    
    Returns:
        VotingClassifier
    """
    estimators = [(name, pipeline) for name, pipeline in pipelines_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    return ensemble


def create_ensemble_stacking(base_estimators, meta_estimator=None):
    """
    Create Stacking Classifier ensemble.
    
    Args:
        base_estimators: List of (name, estimator) tuples
        meta_estimator: Meta-learner (default: LogisticRegression)
    
    Returns:
        StackingClassifier
    """
    from sklearn.linear_model import LogisticRegression
    
    if meta_estimator is None:
        meta_estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_estimator,
        cv=TimeSeriesSplit(n_splits=3),  # Use TimeSeriesSplit for stacking
        n_jobs=-1
    )
    return ensemble


def evaluate_ensemble(ensemble, X_train, y_train, X_test, y_test, name="Ensemble"):
    """
    Evaluate ensemble model.
    
    Returns:
        dict: Evaluation results
    """
    # Train
    ensemble.fit(X_train, y_train)
    
    # Predict
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Metrics
    results = {
        'model': ensemble,
        'name': name,
        'accuracy': (y_pred == y_test).mean(),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results


def create_model_comparison_table(results_dict):
    """
    Create a comparison table of all models.
    
    Args:
        results_dict: Dictionary of {model_name: results}
    
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': results.get('accuracy', 0),
            'Test ROC-AUC': results.get('roc_auc', 0),
            'Test Precision': results.get('precision', 0),
            'Test Recall': results.get('recall', 0),
            'Test F1-Score': results.get('f1_score', 0),
            'CV ROC-AUC Mean': results.get('cv_roc_auc_mean', 0),
            'CV ROC-AUC Std': results.get('cv_roc_auc_std', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Test ROC-AUC', ascending=False)
    return df


def analyze_misclassifications(y_test, y_pred, X_test, merged_test=None, n_samples=20):
    """
    Analyze misclassifications and return examples.
    
    Args:
        y_test: True labels
        y_pred: Predictions
        X_test: Test features
        merged_test: Test data with original features (optional)
        n_samples: Number of misclassified examples to return
    
    Returns:
        DataFrame with misclassified examples
    """
    misclassified = y_test != y_pred
    
    if merged_test is not None and len(merged_test) == len(y_test):
        misclassified_df = merged_test[misclassified].copy()
        misclassified_df['true_label'] = y_test[misclassified]
        misclassified_df['predicted_label'] = y_pred[misclassified]
        
        # Add sentiment information if available
        if 'textblob_sentiment' in misclassified_df.columns:
            return misclassified_df.head(n_samples)
    
    # Fallback: create simple dataframe
    misclassified_indices = np.where(misclassified)[0]
    return pd.DataFrame({
        'index': misclassified_indices[:n_samples],
        'true_label': y_test[misclassified_indices[:n_samples]],
        'predicted_label': y_pred[misclassified_indices[:n_samples]]
    })


def plot_comprehensive_evaluation(results_dict, y_test, output_dir=RESULTS_DIR):
    """
    Create comprehensive evaluation plots for all models.
    
    Args:
        results_dict: Dictionary of {model_name: results}
        y_test: True labels
        output_dir: Output directory for plots
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. ROC Curves for all models
    for model_name, results in results_dict.items():
        y_proba = results['probabilities']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0, 0].plot(fpr, tpr, label=f"{model_name} (AUC={results['roc_auc']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves - All Models')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Model Comparison Bar Chart
    model_names = list(results_dict.keys())
    roc_aucs = [results_dict[m]['roc_auc'] for m in model_names]
    accuracies = [results_dict[m]['accuracy'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    axes[0, 1].bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Model Performance Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Precision-Recall Curves
    for model_name, results in results_dict.items():
        y_proba = results['probabilities']
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[1, 0].plot(recall, precision, label=model_name)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics Comparison
    metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1_score']
    metric_data = {metric: [results_dict[m].get(metric, 0) for m in model_names] 
                   for metric in metrics}
    
    x = np.arange(len(model_names))
    width = 0.15
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2) * width
        axes[1, 1].bar(x + offset, metric_data[metric], width, label=metric, alpha=0.8)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Detailed Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive evaluation plot to: {output_dir / 'comprehensive_evaluation.png'}")


def generate_shap_explanations(model, X_test, feature_names=None, n_samples=100):
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        feature_names: Feature names (optional)
        n_samples: Number of samples to explain (for speed)
    
    Returns:
        SHAP values and explainer
    """
    try:
        import shap
        
        # Sample for speed if dataset is large
        if len(X_test) > n_samples:
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
        else:
            X_sample = X_test
        
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            if hasattr(model, 'feature_importances_') or hasattr(model, 'tree_'):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Linear model or other
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
        else:
            print("⚠️  Model doesn't support SHAP explanations")
            return None, None
        
        return shap_values, explainer
    
    except ImportError:
        print("⚠️  SHAP not installed. Install with: pip install shap")
        return None, None
    except Exception as e:
        print(f"⚠️  Error generating SHAP explanations: {e}")
        return None, None


def plot_shap_summary(shap_values, X_sample, feature_names=None, output_dir=RESULTS_DIR):
    """
    Plot SHAP summary.
    
    Args:
        shap_values: SHAP values
        X_sample: Sample features
        feature_names: Feature names
        output_dir: Output directory
    """
    try:
        import shap
        
        if shap_values is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved SHAP summary plot to: {output_dir / 'shap_summary.png'}")
    
    except Exception as e:
        print(f"⚠️  Error plotting SHAP: {e}")

