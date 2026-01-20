"""
Feature selection module with cross-validation to prevent data leakage
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from .config import RANDOM_STATE


class CVFeatureSelector:
    """
    Feature selector that uses cross-validation to prevent data leakage.
    Feature selection is performed on each training fold separately.
    """
    
    def __init__(self, method='mutual_info', n_features=None, cv_folds=5, random_state=None):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'f_classif', 'rfe', 'select_from_model')
            n_features: Number of features to select (None = auto)
            cv_folds: Number of CV folds
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.random_state = random_state or RANDOM_STATE
        self.selected_features_ = None
        self.selector_ = None
        
    def fit(self, X_train, y_train):
        """
        Fit feature selector using cross-validation.
        Features are selected based only on training folds.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print(f"Selecting features using {self.method} with {self.cv_folds}-fold CV...")
        
        # Determine number of features if not specified
        if self.n_features is None:
            # Select top 50% of features or max 500, whichever is smaller
            self.n_features = min(int(X_train.shape[1] * 0.5), 500)
        
        # Ensure n_features doesn't exceed available features
        self.n_features = min(self.n_features, X_train.shape[1])
        
        # Use cross-validation to select features
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        feature_scores = np.zeros(X_train.shape[1])
        feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        
        # Score features on each fold
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            
            # Select features based on method
            if self.method == 'mutual_info':
                scores, _ = mutual_info_classif(
                    X_fold_train, y_fold_train, 
                    random_state=self.random_state
                )
            elif self.method == 'f_classif':
                scores, _ = f_classif(X_fold_train, y_fold_train)
            else:
                # Default to mutual_info
                scores, _ = mutual_info_classif(
                    X_fold_train, y_fold_train,
                    random_state=self.random_state
                )
            
            feature_scores += scores
        
        # Average scores across folds
        feature_scores /= self.cv_folds
        
        # Select top features
        top_indices = np.argsort(feature_scores)[-self.n_features:][::-1]
        
        if feature_names is not None:
            self.selected_features_ = feature_names[top_indices].tolist()
        else:
            self.selected_features_ = top_indices.tolist()
        
        print(f"✅ Selected {len(self.selected_features_)} features from {X_train.shape[1]} total")
        return self
    
    def transform(self, X):
        """Transform data to selected features only."""
        if self.selected_features_ is None:
            raise ValueError("Feature selector must be fitted first")
        
        if hasattr(X, 'columns'):
            # DataFrame
            return X[self.selected_features_]
        else:
            # Array
            return X[:, self.selected_features_]
    
    def fit_transform(self, X_train, y_train):
        """Fit and transform in one step."""
        self.fit(X_train, y_train)
        return self.transform(X_train)


def select_features_cv(X_train, y_train, method='mutual_info', n_features=None, cv_folds=5):
    """
    Select features using cross-validation to prevent data leakage.
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training target
        method: Selection method ('mutual_info', 'f_classif')
        n_features: Number of features to select (None = auto)
        cv_folds: Number of CV folds
    
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_feature_names, selector)
    """
    selector = CVFeatureSelector(
        method=method,
        n_features=n_features,
        cv_folds=cv_folds,
        random_state=RANDOM_STATE
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    return X_train_selected, selector.selected_features_, selector


def select_features_rfe(X_train, y_train, n_features=None, estimator=None):
    """
    Select features using Recursive Feature Elimination with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_features: Number of features to select
        estimator: Base estimator (default: RandomForestClassifier)
    
    Returns:
        tuple: (X_train_selected, selected_feature_names, selector)
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    if n_features is None:
        n_features = min(int(X_train.shape[1] * 0.5), 500)
    
    print(f"Selecting {n_features} features using RFE...")
    
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=0.1
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[selector.support_].tolist()
    else:
        selected_features = selector.support_
    
    print(f"✅ Selected {len(selected_features)} features using RFE")
    
    return X_train_selected, selected_features, selector


def select_features_from_model(X_train, y_train, estimator=None, threshold='median'):
    """
    Select features using model-based selection with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        estimator: Base estimator (default: RandomForestClassifier)
        threshold: Threshold for feature selection
    
    Returns:
        tuple: (X_train_selected, selected_feature_names, selector)
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    print("Selecting features using model-based selection...")
    
    selector = SelectFromModel(
        estimator=estimator,
        threshold=threshold
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[selector.get_support()].tolist()
    else:
        selected_features = selector.get_support()
    
    print(f"✅ Selected {len(selected_features)} features using model-based selection")
    
    return X_train_selected, selected_features, selector

