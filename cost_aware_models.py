"""
Cost-aware machine learning models for sentiment classification.

This module extends classical ML models with cost-sensitive evaluation and includes:
- SVM (Linear and RBF kernel)
- Multinomial Naive Bayes
- Logistic Regression with class weights
- Random Forest, XGBoost, LightGBM
- All models evaluated with custom cost metrics
"""
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit,
    cross_validate
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from .config import RANDOM_STATE
from .cost_sensitive_evaluation import (
    COST_MATRIX, compute_cost_score, compute_expected_cost,
    make_scorer_cost_sensitive, CLASS_NAMES
)


def train_linear_svm(X_train, y_train, X_test, y_test, class_weight=None, C=1.0):
    """
    Train Linear SVM with cost-aware evaluation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        class_weight: Class weights (dict, 'balanced', or None)
        C: Regularization parameter
    
    Returns:
        dict: Model results including cost metrics
    """
    print("\n[Training] Linear SVM...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(C=C, class_weight=class_weight, random_state=RANDOM_STATE, 
                         max_iter=2000, dual=False))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)
    
    # Get probabilities using decision function (approximate)
    decision_scores = pipeline.decision_function(X_test)
    # Convert decision scores to probabilities using softmax approximation
    y_proba = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    y_proba = y_proba / np.sum(y_proba, axis=1, keepdims=True)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def train_rbf_svm(X_train, y_train, X_test, y_test, class_weight=None, C=1.0, gamma='scale'):
    """Train RBF Kernel SVM with cost-aware evaluation."""
    print("\n[Training] RBF SVM...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C, gamma=gamma, class_weight=class_weight,
                   probability=True, random_state=RANDOM_STATE))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def train_naive_bayes(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train Multinomial Naive Bayes with cost-aware evaluation."""
    print("\n[Training] Multinomial Naive Bayes...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Ensure features are non-negative for Naive Bayes
    # Use MinMaxScaler to scale to [0, 1] instead of StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def train_logistic_regression_weighted(X_train, y_train, X_test, y_test, 
                                      class_weight=None, C=1.0):
    """
    Train Logistic Regression with class weights and cost-aware evaluation.
    
    Args:
        class_weight: Class weights (dict, 'balanced', or None)
    """
    print("\n[Training] Logistic Regression (with class weights)...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, class_weight=class_weight, 
                                  random_state=RANDOM_STATE, max_iter=1000,
                                  multi_class='multinomial', solver='lbfgs', n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred),
        'class_weight': class_weight
    }


def train_random_forest_cost_aware(X_train, y_train, X_test, y_test,
                                   class_weight=None, n_estimators=200, max_depth=15):
    """Train Random Forest with class weights and cost-aware evaluation."""
    print("\n[Training] Random Forest (cost-aware)...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def train_xgboost_cost_aware(X_train, y_train, X_test, y_test,
                             n_estimators=200, max_depth=5, learning_rate=0.1):
    """Train XGBoost with cost-aware evaluation."""
    print("\n[Training] XGBoost (cost-aware)...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Convert to numpy arrays if DataFrames (XGBoost works better with arrays)
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_array, y_train_encoded)
    y_pred = model.predict(X_test_array)
    y_proba = model.predict_proba(X_test_array)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def train_lightgbm_cost_aware(X_train, y_train, X_test, y_test,
                              n_estimators=200, max_depth=5, learning_rate=0.1):
    """Train LightGBM with cost-aware evaluation."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required for train_lightgbm_cost_aware. "
                         "Install it with: pip install lightgbm")
    
    print("\n[Training] LightGBM (cost-aware)...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='multiclass',
        num_class=3,
        verbose=-1
    )
    
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    cost = compute_cost_score(y_test_encoded, y_pred)
    expected_cost = compute_expected_cost(y_test_encoded, y_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
    }


def tune_hyperparameters_cost_aware(model, param_grid, X_train, y_train,
                                   cv_folds=5, n_iter=50, use_random_search=True):
    """
    Tune hyperparameters using cost-aware scoring.
    
    Args:
        model: Base model or pipeline
        param_grid: Parameter grid for tuning
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        n_iter: Number of iterations for random search
        use_random_search: If True, use RandomizedSearchCV, else GridSearchCV
    
    Returns:
        Best model with tuned hyperparameters
    """
    print(f"\n🔍 Tuning hyperparameters (cost-aware)...")
    
    # Encode labels if needed
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
    else:
        y_train_encoded = y_train
    
    cv = TimeSeriesSplit(n_splits=cv_folds)
    cost_scorer = make_scorer_cost_sensitive(use_proba=False)
    
    if use_random_search:
        search = RandomizedSearchCV(
            model, param_grid, cv=cv, scoring=cost_scorer,
            n_iter=n_iter, random_state=RANDOM_STATE, n_jobs=-1, verbose=1
        )
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=cost_scorer,
            n_jobs=-1, verbose=1
        )
    
    search.fit(X_train, y_train_encoded)
    
    print(f"✅ Best parameters: {search.best_params_}")
    print(f"✅ Best cost score: {-search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_


def experiment_class_weights(X_train, y_train, X_test, y_test):
    """
    Experiment with different class weight settings vs custom cost scorer.
    
    Returns:
        dict: Results for different class weight configurations
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Class Weights vs Custom Cost Scorer")
    print("="*60)
    
    results = {}
    
    # Different class weight configurations
    weight_configs = {
        'uniform': None,
        'balanced': 'balanced',
        'cost_aware': {0: 5.0, 1: 1.0, 2: 5.0},  # Emphasize negative/positive errors
        'cost_aware_2': {0: 3.0, 1: 1.0, 2: 3.0}
    }
    
    for config_name, class_weight in weight_configs.items():
        print(f"\n--- Configuration: {config_name} ---")
        result = train_logistic_regression_weighted(
            X_train, y_train, X_test, y_test, class_weight=class_weight
        )
        results[config_name] = result
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Cost: {result['cost']:.4f}")
        print(f"  Expected Cost: {result['expected_cost']:.4f}")
    
    return results


def train_all_cost_aware_models(X_train, y_train, X_test, y_test):
    """
    Train all cost-aware models and compare performance.
    
    Returns:
        dict: Results for all models
    """
    print("\n" + "="*60)
    print("TRAINING ALL COST-AWARE MODELS")
    print("="*60)
    
    results = {}
    
    # 1. Linear SVM
    results['Linear SVM'] = train_linear_svm(X_train, y_train, X_test, y_test)
    
    # 2. RBF SVM
    results['RBF SVM'] = train_rbf_svm(X_train, y_train, X_test, y_test)
    
    # 3. Naive Bayes
    results['Naive Bayes'] = train_naive_bayes(X_train, y_train, X_test, y_test)
    
    # 4. Logistic Regression (balanced weights)
    results['Logistic Regression'] = train_logistic_regression_weighted(
        X_train, y_train, X_test, y_test, class_weight='balanced'
    )
    
    # 5. Random Forest
    results['Random Forest'] = train_random_forest_cost_aware(
        X_train, y_train, X_test, y_test
    )
    
    # 6. XGBoost
    results['XGBoost'] = train_xgboost_cost_aware(
        X_train, y_train, X_test, y_test
    )
    
    # 7. LightGBM (if available)
    try:
        results['LightGBM'] = train_lightgbm_cost_aware(
            X_train, y_train, X_test, y_test
        )
    except ImportError:
        print("[WARNING] LightGBM not available, skipping...")
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON (Cost-Aware)")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Cost':<10} {'Expected Cost':<15} {'F1-Macro':<10}")
    print("-" * 70)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']:<10.4f} {result['cost']:<10.4f} "
              f"{result['expected_cost']:<15.4f} {result['f1_macro']:<10.4f}")
    
    # Find best model (lowest cost)
    best_model_name = min(results.keys(), key=lambda x: results[x]['cost'])
    print(f"\n[Best Model] (Lowest Cost): {best_model_name}")
    print(f"   Cost: {results[best_model_name]['cost']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return results

