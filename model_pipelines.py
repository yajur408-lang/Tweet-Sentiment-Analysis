"""
Model pipeline module with scikit-learn Pipeline and ColumnTransformer
for safe preprocessing and time series cross-validation
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    roc_curve, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import (
    RANDOM_STATE, TEST_SIZE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE
)


def create_time_series_split(merged_data, test_size=0.2):
    """
    Split data into train and test sets ensuring test set is from a later period.
    For time series data, this prevents lookahead bias.
    
    Args:
        merged_data: Merged dataframe with 'date' column
        test_size: Proportion of data to use for testing
    
    Returns:
        tuple: (train_indices, test_indices)
    """
    # Ensure data is sorted by date
    if 'date' in merged_data.columns:
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
    else:
        raise ValueError("Data must have a 'date' column for time series splitting")
    
    # Calculate split point
    n_samples = len(merged_data)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = merged_data.index[:split_idx].tolist()
    test_indices = merged_data.index[split_idx:].tolist()
    
    print(f"✅ Time series split:")
    print(f"   Training set: {len(train_indices):,} samples (earlier period)")
    print(f"   Test set: {len(test_indices):,} samples (later period)")
    
    if 'date' in merged_data.columns:
        train_dates = merged_data.loc[train_indices, 'date']
        test_dates = merged_data.loc[test_indices, 'date']
        print(f"   Training date range: {train_dates.min()} to {train_dates.max()}")
        print(f"   Test date range: {test_dates.min()} to {test_dates.max()}")
        print(f"   🔒 No lookahead bias: Test set is from later period")
    
    return train_indices, test_indices


def create_preprocessing_pipeline(feature_names=None, use_scaling=True, scaler_type='standard'):
    """
    Create preprocessing pipeline using ColumnTransformer.
    This ensures all transformations are applied safely.
    
    Args:
        feature_names: List of feature names (for DataFrame support)
        use_scaling: Whether to apply scaling
        scaler_type: Type of scaler ('standard' or 'robust')
    
    Returns:
        ColumnTransformer or scaler for preprocessing
    """
    if not use_scaling:
        # Return identity transformer
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(func=lambda X: X, inverse_func=lambda X: X)
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # For numeric features, we can use scaler directly
    # ColumnTransformer is useful when we have mixed types, but for simplicity
    # and since all our features are numeric, we'll use scaler directly
    return scaler


def create_logistic_regression_pipeline(use_scaling=True, scaler_type='standard', 
                                       feature_selection=None, C=1.0, max_iter=1000):
    """
    Create Logistic Regression pipeline with preprocessing.
    
    Args:
        use_scaling: Whether to apply scaling
        scaler_type: Type of scaler ('standard' or 'robust')
        feature_selection: Feature selection step (optional)
        C: Regularization parameter
        max_iter: Maximum iterations
    
    Returns:
        Pipeline object
    """
    steps = []
    
    # Preprocessing
    if use_scaling:
        preprocessor = create_preprocessing_pipeline(use_scaling=use_scaling, scaler_type=scaler_type)
        steps.append(('scaler', preprocessor))
    
    # Feature selection (if provided)
    if feature_selection is not None:
        steps.append(('feature_selection', feature_selection))
    
    # Model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        solver='lbfgs'
    )
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def create_random_forest_pipeline(use_scaling=False, feature_selection=None,
                                  n_estimators=None, max_depth=None):
    """
    Create Random Forest pipeline with preprocessing.
    Note: Random Forest typically doesn't need scaling.
    
    Args:
        use_scaling: Whether to apply scaling (usually False for RF)
        feature_selection: Feature selection step (optional)
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    
    Returns:
        Pipeline object
    """
    steps = []
    
    # Preprocessing (optional for RF)
    if use_scaling:
        preprocessor = create_preprocessing_pipeline(use_scaling=True)
        steps.append(('scaler', preprocessor))
    
    # Feature selection (if provided)
    if feature_selection is not None:
        steps.append(('feature_selection', feature_selection))
    
    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators or RF_N_ESTIMATORS,
        max_depth=max_depth or RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def create_xgboost_pipeline(use_scaling=False, feature_selection=None,
                            n_estimators=None, max_depth=None, learning_rate=None):
    """
    Create XGBoost pipeline with preprocessing.
    Note: XGBoost typically doesn't need scaling.
    
    Args:
        use_scaling: Whether to apply scaling (usually False for XGBoost)
        feature_selection: Feature selection step (optional)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
    
    Returns:
        Pipeline object
    """
    steps = []
    
    # Preprocessing (optional for XGBoost)
    if use_scaling:
        preprocessor = create_preprocessing_pipeline(use_scaling=True)
        steps.append(('scaler', preprocessor))
    
    # Feature selection (if provided)
    if feature_selection is not None:
        steps.append(('feature_selection', feature_selection))
    
    # Model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators or XGB_N_ESTIMATORS,
        max_depth=max_depth or XGB_MAX_DEPTH,
        learning_rate=learning_rate or XGB_LEARNING_RATE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def train_model_with_cv(pipeline, X_train, y_train, cv_folds=5, 
                        use_time_series_cv=True, scoring=['accuracy', 'roc_auc']):
    """
    Train model with cross-validation using TimeSeriesSplit.
    All preprocessing is wrapped inside the CV loop to prevent leakage.
    
    Args:
        pipeline: sklearn Pipeline object
        X_train: Training features
        y_train: Training target
        cv_folds: Number of CV folds
        use_time_series_cv: Whether to use TimeSeriesSplit (True) or KFold (False)
        scoring: List of scoring metrics
    
    Returns:
        dict: Cross-validation results
    """
    if use_time_series_cv:
        cv = TimeSeriesSplit(n_splits=cv_folds)
        print(f"🔒 Using TimeSeriesSplit ({cv_folds} folds) - training data always precedes test data")
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        print(f"Using KFold ({cv_folds} folds)")
    
    # Perform cross-validation
    # All preprocessing (scaling, feature selection) happens inside CV loop
    print(f"Running cross-validation ({cv_folds} folds)...")
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Print CV results
    print(f"\n📊 Cross-Validation Results:")
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        print(f"   {metric.upper()}:")
        print(f"     Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
        print(f"     Test:  {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")
    
    return cv_results


def train_all_models_pipelines(X_train, X_test, y_train, y_test, 
                               use_feature_selection=False, feature_selection_params=None,
                               cv_folds=5, use_time_series_cv=True):
    """
    Train all three models (Logistic Regression, Random Forest, XGBoost) using pipelines.
    All preprocessing is safely wrapped in pipelines.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        use_feature_selection: Whether to use feature selection
        feature_selection_params: Parameters for feature selection
        cv_folds: Number of CV folds
        use_time_series_cv: Whether to use TimeSeriesSplit
    
    Returns:
        dict: Results for all models
    """
    results = {}
    
    # Feature selection (if enabled)
    feature_selection_step = None
    if use_feature_selection and feature_selection_params:
        from .feature_selection import CVFeatureSelector
        feature_selection_step = CVFeatureSelector(
            method=feature_selection_params.get('method', 'mutual_info'),
            n_features=feature_selection_params.get('n_features', None),
            cv_folds=feature_selection_params.get('cv_folds', 5),
            random_state=RANDOM_STATE
        )
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    with tqdm(total=3, desc="Setting up Logistic Regression") as pbar:
        lr_pipeline = create_logistic_regression_pipeline(
            use_scaling=True,
            scaler_type='standard',
            feature_selection=feature_selection_step
        )
        pbar.update(1)
    
    # Cross-validation
    lr_cv_results = train_model_with_cv(
        lr_pipeline, X_train, y_train,
        cv_folds=cv_folds,
        use_time_series_cv=use_time_series_cv
    )
    
    # Train on full training set
    with tqdm(total=1, desc="Training Logistic Regression on full dataset") as pbar:
        lr_pipeline.fit(X_train, y_train)
        pbar.update(1)
    
    with tqdm(total=2, desc="Making predictions") as pbar:
        lr_pred = lr_pipeline.predict(X_test)
        pbar.update(1)
        lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
        pbar.update(1)
    
    results['Logistic Regression'] = {
        'pipeline': lr_pipeline,
        'model': lr_pipeline.named_steps['classifier'],
        'cv_results': lr_cv_results,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'cv_accuracy_mean': lr_cv_results['test_accuracy'].mean(),
        'cv_roc_auc_mean': lr_cv_results['test_roc_auc'].mean()
    }
    
    # 2. Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    with tqdm(total=1, desc="Setting up Random Forest") as pbar:
        rf_pipeline = create_random_forest_pipeline(
            use_scaling=False,  # RF doesn't need scaling
            feature_selection=feature_selection_step
        )
        pbar.update(1)
    
    # Cross-validation
    rf_cv_results = train_model_with_cv(
        rf_pipeline, X_train, y_train,
        cv_folds=cv_folds,
        use_time_series_cv=use_time_series_cv
    )
    
    # Train on full training set
    with tqdm(total=1, desc="Training Random Forest on full dataset") as pbar:
        rf_pipeline.fit(X_train, y_train)
        pbar.update(1)
    
    with tqdm(total=2, desc="Making predictions") as pbar:
        rf_pred = rf_pipeline.predict(X_test)
        pbar.update(1)
        rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
        pbar.update(1)
    
    results['Random Forest'] = {
        'pipeline': rf_pipeline,
        'model': rf_pipeline.named_steps['classifier'],
        'cv_results': rf_cv_results,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'cv_accuracy_mean': rf_cv_results['test_accuracy'].mean(),
        'cv_roc_auc_mean': rf_cv_results['test_roc_auc'].mean()
    }
    
    # 3. XGBoost
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    with tqdm(total=1, desc="Setting up XGBoost") as pbar:
        xgb_pipeline = create_xgboost_pipeline(
            use_scaling=False,  # XGBoost doesn't need scaling
            feature_selection=feature_selection_step
        )
        pbar.update(1)
    
    # Cross-validation
    xgb_cv_results = train_model_with_cv(
        xgb_pipeline, X_train, y_train,
        cv_folds=cv_folds,
        use_time_series_cv=use_time_series_cv
    )
    
    # Train on full training set
    with tqdm(total=1, desc="Training XGBoost on full dataset") as pbar:
        xgb_pipeline.fit(X_train, y_train)
        pbar.update(1)
    
    with tqdm(total=2, desc="Making predictions") as pbar:
        xgb_pred = xgb_pipeline.predict(X_test)
        pbar.update(1)
        xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
        pbar.update(1)
    
    results['XGBoost'] = {
        'pipeline': xgb_pipeline,
        'model': xgb_pipeline.named_steps['classifier'],
        'cv_results': xgb_cv_results,
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'roc_auc': roc_auc_score(y_test, xgb_proba),
        'cv_accuracy_mean': xgb_cv_results['test_accuracy'].mean(),
        'cv_roc_auc_mean': xgb_cv_results['test_roc_auc'].mean()
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'CV Accuracy':<15} {'CV ROC-AUC':<15} {'Test Accuracy':<15} {'Test ROC-AUC':<15}")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['cv_accuracy_mean']:<15.4f} {metrics['cv_roc_auc_mean']:<15.4f} "
              f"{metrics['accuracy']:<15.4f} {metrics['roc_auc']:<15.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"\n🏆 Best Model (Test ROC-AUC): {best_model_name} ({results[best_model_name]['roc_auc']:.4f})")
    
    return results, best_model_name


def evaluate_model_detailed(model_results, model_name, y_test, X_test=None):
    """
    Perform detailed evaluation of a model including classification report.
    
    Args:
        model_results: Dictionary with model results
        model_name: Name of the model
        y_test: True labels
        X_test: Test features (optional, for feature importance)
    
    Returns:
        dict: Detailed evaluation metrics
    """
    y_pred = model_results['predictions']
    y_proba = model_results['probabilities']
    
    print("\n" + "="*60)
    print(f"DETAILED EVALUATION: {model_name}")
    print("="*60)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        Up    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 Additional Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:   {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC:  {model_results['roc_auc']:.4f}")
    
    evaluation = {
        'classification_report': classification_report(y_test, y_pred, target_names=['Down', 'Up'], output_dict=True),
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': model_results['roc_auc'],
        'accuracy': model_results['accuracy']
    }
    
    return evaluation

