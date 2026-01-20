"""
Cost-sensitive evaluation module for sentiment analysis.

Defines a cost matrix where:
- negative ↔ positive = very bad (cost 5)
- negative ↔ neutral or neutral ↔ positive = less bad (cost 1)
- same class = no cost (0)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow (optional for deep learning)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from .config import RESULTS_DIR

# Cost matrix definition
# Row = true class, Column = predicted class
# Format: [negative, neutral, positive]
COST_MATRIX = np.array([
    [0., 1., 5.],  # True: negative
    [1., 0., 1.],  # True: neutral
    [5., 1., 0.]   # True: positive
], dtype=np.float32)

CLASS_NAMES = ['negative', 'neutral', 'positive']
CLASS_TO_IDX = {'negative': 0, 'neutral': 1, 'positive': 2}
IDX_TO_CLASS = {0: 'negative', 1: 'neutral', 2: 'positive'}

# TensorFlow version for deep learning (only if TensorFlow is available)
COST_MATRIX_TF = None
if TF_AVAILABLE:
    COST_MATRIX_TF = tf.constant(COST_MATRIX, dtype=tf.float32)


def compute_cost_score(y_true: np.ndarray, y_pred: np.ndarray, 
                       cost_matrix: np.ndarray = None) -> float:
    """
    Compute total cost score given true labels and predictions.
    
    Args:
        y_true: True labels (can be strings or integers)
        y_pred: Predicted labels (can be strings or integers)
        cost_matrix: Cost matrix (default: COST_MATRIX)
    
    Returns:
        float: Total cost
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    # Convert string labels to integers if needed
    if isinstance(y_true[0], str):
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
    
    # Ensure labels are in [0, 1, 2] range
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Compute cost for each prediction
    total_cost = 0.0
    for true_label, pred_label in zip(y_true, y_pred):
        total_cost += cost_matrix[true_label, pred_label]
    
    return total_cost


def compute_expected_cost(y_true: np.ndarray, y_proba: np.ndarray,
                         cost_matrix: np.ndarray = None) -> float:
    """
    Compute expected cost using probability predictions.
    
    Args:
        y_true: True labels (integers or strings)
        y_proba: Prediction probabilities (n_samples, n_classes)
        cost_matrix: Cost matrix (default: COST_MATRIX)
    
    Returns:
        float: Expected cost
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    # Convert string labels to integers if needed
    if isinstance(y_true[0], str):
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        y_true = le.transform(y_true)
    
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba)
    
    if y_proba.shape[1] != cost_matrix.shape[0]:
        raise ValueError(f"y_proba has {y_proba.shape[1]} classes, but cost_matrix has {cost_matrix.shape[0]} classes")
    
    # Compute expected cost: for each sample, sum(prob * cost_row)
    expected_costs = []
    for i, true_label in enumerate(y_true):
        cost_row = cost_matrix[true_label, :]
        expected_cost = np.sum(y_proba[i] * cost_row)
        expected_costs.append(expected_cost)
    
    return np.mean(expected_costs)


def make_scorer_cost_sensitive(cost_matrix: np.ndarray = None, use_proba: bool = False):
    """
    Create a sklearn-compatible scorer for cost-sensitive evaluation.
    
    Args:
        cost_matrix: Cost matrix (default: COST_MATRIX)
        use_proba: If True, use probabilities (expected cost). If False, use predictions (actual cost).
    
    Returns:
        callable: Scorer function
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    def cost_scorer(y_true, y_pred_or_proba):
        """Scorer that returns negative cost (higher is better)."""
        if use_proba:
            # y_pred_or_proba is actually probabilities
            cost = compute_expected_cost(y_true, y_pred_or_proba, cost_matrix)
        else:
            # y_pred_or_proba is predictions
            cost = compute_cost_score(y_true, y_pred_or_proba, cost_matrix)
        
        # Return negative cost so that higher scores are better
        return -cost
    
    return cost_scorer


def custom_sentiment_loss(y_true, y_pred):
    """
    Custom loss function for TensorFlow/Keras that uses the cost matrix.
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch, 3)
        y_pred: Predicted probabilities, shape (batch, 3)
    
    Returns:
        Tensor: Mean expected cost per batch
    
    Raises:
        ImportError: If TensorFlow is not available
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for custom_sentiment_loss. "
                         "Install it with: pip install tensorflow")
    
    # Convert one-hot to integer class indices
    true_idx = tf.argmax(y_true, axis=1)  # (batch,)
    
    # Gather cost row for each true label
    cost_rows = tf.gather(COST_MATRIX_TF, true_idx)  # (batch, 3)
    
    # Expected cost = sum(probs * cost_row) for each sample
    expected_cost = tf.reduce_sum(y_pred * cost_rows, axis=1)  # (batch,)
    
    return tf.reduce_mean(expected_cost)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          thresholds: np.ndarray = None,
                          cost_matrix: np.ndarray = None) -> Dict[str, Any]:
    """
    Find optimal threshold for cost-sensitive decisions.
    
    Sweeps different thresholds for predicting positive class and selects
    the one that minimizes expected cost.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities (n_samples, n_classes)
        thresholds: List of thresholds to try (default: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        cost_matrix: Cost matrix (default: COST_MATRIX)
    
    Returns:
        dict: Results including optimal threshold and costs
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    if thresholds is None:
        thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Convert string labels to integers if needed
    if isinstance(y_true[0], str):
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        y_true_encoded = le.transform(y_true)
    else:
        y_true_encoded = y_true
        le = None
    
    y_true_encoded = np.asarray(y_true_encoded, dtype=int)
    y_proba = np.asarray(y_proba)
    
    if y_proba.shape[1] != 3:
        raise ValueError("y_proba must have 3 classes (negative, neutral, positive)")
    
    results = []
    
    for threshold in thresholds:
        # Create predictions based on threshold
        # Only predict "positive" if p(positive) > threshold
        # Otherwise, use argmax of remaining classes
        y_pred = []
        for i in range(len(y_proba)):
            if y_proba[i, 2] > threshold:  # p(positive) > threshold
                y_pred.append(2)  # positive
            else:
                # Predict based on negative vs neutral
                if y_proba[i, 0] > y_proba[i, 1]:
                    y_pred.append(0)  # negative
                else:
                    y_pred.append(1)  # neutral
        
        y_pred = np.array(y_pred)
        
        # Compute cost
        actual_cost = compute_cost_score(y_true_encoded, y_pred, cost_matrix)
        expected_cost = compute_expected_cost(y_true_encoded, y_proba, cost_matrix)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred)
        
        results.append({
            'threshold': threshold,
            'actual_cost': actual_cost,
            'expected_cost': expected_cost,
            'confusion_matrix': cm,
            'predictions': y_pred.copy()
        })
    
    # Find optimal threshold (minimum cost)
    optimal_idx = np.argmin([r['actual_cost'] for r in results])
    optimal_result = results[optimal_idx]
    
    return {
        'optimal_threshold': optimal_result['threshold'],
        'optimal_cost': optimal_result['actual_cost'],
        'optimal_predictions': optimal_result['predictions'],
        'optimal_confusion_matrix': optimal_result['confusion_matrix'],
        'all_results': results
    }


def visualize_cost_matrix(cost_matrix: np.ndarray = None, save_path: Optional[str] = None):
    """Visualize the cost matrix as a heatmap."""
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cost_matrix, annot=True, fmt='.1f', cmap='Reds', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Cost'})
    plt.title('Cost Matrix for Sentiment Classification\n(Row=True, Column=Predicted)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved cost matrix visualization to: {save_path}")
    
    return plt.gcf()


def analyze_expensive_mistakes(y_true: np.ndarray, y_pred: np.ndarray, 
                              texts: Optional[List[str]] = None,
                              cost_matrix: np.ndarray = None,
                              n_examples: int = 20) -> Dict[str, Any]:
    """
    Analyze expensive mistakes (negative→positive and positive→negative).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Optional list of text samples (tweets)
        cost_matrix: Cost matrix (default: COST_MATRIX)
        n_examples: Number of examples to show per mistake type
    
    Returns:
        dict: Analysis results with examples
    """
    if cost_matrix is None:
        cost_matrix = COST_MATRIX
    
    # Convert string labels to integers if needed
    # Handle different cases: both strings, both ints, or mixed
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    # Check if y_true contains strings
    if len(y_true_arr) > 0 and isinstance(y_true_arr[0], str):
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        y_true_encoded = le.transform(y_true_arr)
        # If y_pred is also strings, transform it; otherwise it's already integers
        if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
            y_pred_encoded = le.transform(y_pred_arr)
        else:
            y_pred_encoded = np.asarray(y_pred_arr, dtype=int)
    else:
        # Both are already integers (or numeric)
        y_true_encoded = np.asarray(y_true_arr, dtype=int)
        y_pred_encoded = np.asarray(y_pred_arr, dtype=int)
        le = None
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    # Find expensive mistakes
    # negative→positive (cost 5)
    neg_to_pos_indices = np.where((y_true_encoded == 0) & (y_pred_encoded == 2))[0]
    # positive→negative (cost 5)
    pos_to_neg_indices = np.where((y_true_encoded == 2) & (y_pred_encoded == 0))[0]
    # neutral→positive (cost 1)
    neu_to_pos_indices = np.where((y_true_encoded == 1) & (y_pred_encoded == 2))[0]
    # neutral→negative (cost 1)
    neu_to_neg_indices = np.where((y_true_encoded == 1) & (y_pred_encoded == 0))[0]
    
    analysis = {
        'confusion_matrix': cm,
        'total_cost': compute_cost_score(y_true_encoded, y_pred_encoded, cost_matrix),
        'mistake_counts': {
            'negative_to_positive': len(neg_to_pos_indices),
            'positive_to_negative': len(pos_to_neg_indices),
            'neutral_to_positive': len(neu_to_pos_indices),
            'neutral_to_negative': len(neu_to_neg_indices)
        },
        'examples': {}
    }
    
    # Collect examples if texts are provided
    if texts is not None:
        if len(neg_to_pos_indices) > 0:
            sample_indices = neg_to_pos_indices[:n_examples]
            analysis['examples']['negative_to_positive'] = [
                {'text': texts[i], 'true': CLASS_NAMES[y_true_encoded[i]], 
                 'pred': CLASS_NAMES[y_pred_encoded[i]]}
                for i in sample_indices
            ]
        
        if len(pos_to_neg_indices) > 0:
            sample_indices = pos_to_neg_indices[:n_examples]
            analysis['examples']['positive_to_negative'] = [
                {'text': texts[i], 'true': CLASS_NAMES[y_true_encoded[i]], 
                 'pred': CLASS_NAMES[y_pred_encoded[i]]}
                for i in sample_indices
            ]
    
    return analysis


def print_cost_analysis(analysis: Dict[str, Any]):
    """Print formatted cost analysis results."""
    print("\n" + "="*60)
    print("COST ANALYSIS")
    print("="*60)
    
    print(f"\nTotal Cost: {analysis['total_cost']:.2f}")
    print(f"\nMistake Counts:")
    for mistake_type, count in analysis['mistake_counts'].items():
        print(f"  {mistake_type.replace('_', ' ').title()}: {count}")
    
    print(f"\nConfusion Matrix:")
    cm = analysis['confusion_matrix']
    print(f"                Predicted")
    print(f"              Neg  Neu  Pos")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"True {class_name:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}")
    
    # Print examples
    if 'examples' in analysis and analysis['examples']:
        print(f"\n{'='*60}")
        print("EXPENSIVE MISTAKE EXAMPLES")
        print("="*60)
        
        for mistake_type, examples in analysis['examples'].items():
            print(f"\n{mistake_type.replace('_', ' ').title()} ({len(examples)} examples):")
            print("-" * 60)
            for idx, ex in enumerate(examples[:10], 1):  # Show first 10
                print(f"\nExample {idx}:")
                print(f"  True: {ex['true']}")
                print(f"  Pred: {ex['pred']}")
                print(f"  Text: {ex['text'][:200]}..." if len(ex['text']) > 200 else f"  Text: {ex['text']}")


def create_cost_aware_confusion_matrix_plot(analysis: Dict[str, Any], 
                                           save_path: Optional[str] = None):
    """Create an annotated confusion matrix plot with cost information."""
    cm = analysis['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotation matrix with cost information
    annot = []
    for i in range(len(CLASS_NAMES)):
        row = []
        for j in range(len(CLASS_NAMES)):
            count = cm[i, j]
            cost = COST_MATRIX[i, j]
            if count > 0:
                row.append(f"{count}\n(cost: {cost})")
            else:
                row.append(f"{count}")
        annot.append(row)
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='YlOrRd', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    plt.title(f'Cost-Aware Confusion Matrix\nTotal Cost: {analysis["total_cost"]:.2f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved confusion matrix to: {save_path}")
    
    return fig

