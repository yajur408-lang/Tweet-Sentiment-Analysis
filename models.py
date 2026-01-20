"""
Machine learning models module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from .config import (
    RANDOM_STATE, TEST_SIZE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    GB_N_ESTIMATORS, GB_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE
)


def prepare_features(merged_data, w2v_embeddings=None, tfidf_embeddings=None):
    """
    Prepare feature matrix for machine learning.
    Combines sentiment features, technical indicators, and embeddings.
    
    Args:
        merged_data: Dataframe with sentiment and technical features
        w2v_embeddings: DataFrame with Word2Vec embeddings (optional)
        tfidf_embeddings: DataFrame with TF-IDF embeddings (optional)
    
    Returns:
        tuple: (X_full, y) feature matrix and target
    """
    print("Preparing features for machine learning...")
    
    feature_list = []
    
    # Word2Vec embeddings (if provided)
    if w2v_embeddings is not None:
        feature_list.append(w2v_embeddings)
        print(f"  - Added Word2Vec embeddings: {w2v_embeddings.shape[1]} features")
    
    # TF-IDF embeddings (if provided)
    if tfidf_embeddings is not None:
        feature_list.append(tfidf_embeddings)
        print(f"  - Added TF-IDF embeddings: {tfidf_embeddings.shape[1]} features")
    
    # Sentiment and text features
    sentiment_features = [
        'tb_polarity', 'vader_compound', 'tweet_length', 'word_count', 
        'hashtag_count', 'count_buy', 'count_sell', 'count_bullish', 'count_bearish'
    ]
    sentiment_features = [f for f in sentiment_features if f in merged_data.columns]
    if sentiment_features:
        X_sentiment = merged_data[sentiment_features]
        feature_list.append(X_sentiment)
        print(f"  - Added sentiment features: {len(sentiment_features)} features")
    
    # Technical indicators
    tech_features = ['ma_5', 'ma_10', 'ma_20', 'price_change_pct', 'rsi', 'volatility']
    tech_features = [f for f in tech_features if f in merged_data.columns]
    if tech_features:
        X_tech = merged_data[tech_features]
        feature_list.append(X_tech)
        print(f"  - Added technical indicators: {len(tech_features)} features")
    
    # Combine all features
    if feature_list:
        X_full = pd.concat(feature_list, axis=1)
    else:
        raise ValueError("No features available! Check your data.")
    
    y = merged_data['target']
    
    # Remove any remaining NaN values
    nan_mask = X_full.isna().any(axis=1) | y.isna()
    X_full = X_full[~nan_mask]
    y = y[~nan_mask]
    
    print(f"[OK] Feature matrix shape: {X_full.shape}")
    print(f"[OK] Target distribution: {y.value_counts().to_dict()}")
    
    return X_full, y


def train_test_split_data(X_full, y):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[OK] Training set: {X_train.shape[0]} samples")
    print(f"[OK] Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model."""
    print("🚀 Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("⚡ Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all models and compare performance.
    
    Returns:
        dict: Dictionary with model results
    """
    results = {}
    
    results['Random Forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    results['Gradient Boosting'] = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['XGBoost'] = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
    
    return results, best_model_name


def evaluate_model(model_results, model_name, y_test):
    """Print detailed evaluation of a model."""
    y_pred = model_results['predictions']
    y_proba = model_results['probabilities']
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    return y_pred, y_proba


def apply_feature_selection(X_train, X_test, y_train, use_selection=True, method='mutual_info', n_features=None):
    """
    Apply feature selection using cross-validation to prevent data leakage.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        use_selection: Whether to apply feature selection
        method: Selection method ('mutual_info', 'f_classif', 'rfe', 'select_from_model')
        n_features: Number of features to select (None = auto)
    
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_features, selector)
    """
    if not use_selection:
        print("⚠️  Feature selection disabled - using all features")
        return X_train, X_test, list(X_train.columns) if hasattr(X_train, 'columns') else None, None
    
    from .feature_selection import (
        select_features_cv, select_features_rfe, select_features_from_model
    )
    
    print("\n" + "="*60)
    print("FEATURE SELECTION (Cross-Validation Based)")
    print("="*60)
    
    if method == 'rfe':
        X_train_selected, selected_features, selector = select_features_rfe(
            X_train, y_train, n_features=n_features
        )
        X_test_selected = selector.transform(X_test)
    elif method == 'select_from_model':
        X_train_selected, selected_features, selector = select_features_from_model(
            X_train, y_train
        )
        X_test_selected = selector.transform(X_test)
    else:
        # Default: CV-based selection
        X_train_selected, selected_features, selector = select_features_cv(
            X_train, y_train, method=method, n_features=n_features
        )
        X_test_selected = selector.transform(X_test)
    
    print(f"[OK] Feature selection complete: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")
    print(f"🔒 Selection performed using cross-validation (no test data leakage)")
    
    return X_train_selected, X_test_selected, selected_features, selector

