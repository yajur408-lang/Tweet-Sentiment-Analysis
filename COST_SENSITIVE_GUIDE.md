# Cost-Sensitive Sentiment Analysis Guide

## Overview

This guide explains the cost-sensitive evaluation system for sentiment classification. The system implements asymmetric costs for different types of misclassifications, where:

- **negative ↔ positive** = very bad (cost 5)
- **negative ↔ neutral** or **neutral ↔ positive** = less bad (cost 1)
- **same class** = no cost (0)

## Cost Matrix

The cost matrix is defined as:

```
                    Neg   Neutral  Positive
Negative            0      1        5
Neutral             1      0        1
Positive            5      1        0
```

This matrix reflects that confusing extreme sentiments (negative vs positive) is much worse than confusing them with neutral.

## Implementation Modules

### 1. `cost_sensitive_evaluation.py`

Core module for cost-aware evaluation:

- **Cost Matrix Definition**: `COST_MATRIX` and `COST_MATRIX_TF` (TensorFlow version)
- **Cost Scoring**: `compute_cost_score()` - computes actual cost from predictions
- **Expected Cost**: `compute_expected_cost()` - computes expected cost from probabilities
- **Custom Loss Function**: `custom_sentiment_loss()` - TensorFlow/Keras loss function
- **Threshold Tuning**: `find_optimal_threshold()` - finds optimal decision thresholds
- **Mistake Analysis**: `analyze_expensive_mistakes()` - identifies and analyzes costly errors
- **Visualization**: Functions to visualize cost matrix and confusion matrix

### 2. `cost_aware_models.py`

Classical machine learning models with cost-aware evaluation:

- **Linear SVM**: `train_linear_svm()` - Linear Support Vector Machine
- **RBF SVM**: `train_rbf_svm()` - RBF kernel Support Vector Machine
- **Naive Bayes**: `train_naive_bayes()` - Multinomial Naive Bayes
- **Logistic Regression**: `train_logistic_regression_weighted()` - with class weights
- **Random Forest**: `train_random_forest_cost_aware()` - with cost-aware evaluation
- **XGBoost**: `train_xgboost_cost_aware()` - gradient boosting with cost metrics
- **LightGBM**: `train_lightgbm_cost_aware()` - light gradient boosting (optional)
- **Class Weight Experiments**: `experiment_class_weights()` - compares different class weight strategies
- **Hyperparameter Tuning**: `tune_hyperparameters_cost_aware()` - cost-aware hyperparameter tuning

### 3. `deep_learning_models.py`

Deep learning models with custom cost-aware loss:

- **Feed-forward NN**: `train_feedforward_nn_on_embeddings()` - dense network on averaged embeddings
- **1D CNN**: `build_cnn_1d()` - convolutional network for text sequences
- **Simple RNN**: `build_simple_rnn()` - vanilla RNN (for learning purposes)

All deep learning models can use the `custom_sentiment_loss()` function instead of standard cross-entropy.

### 4. `train_cost_sensitive.py`

Comprehensive training script that demonstrates:

1. Cost matrix definition and visualization
2. Training multiple classical ML models with cost-aware evaluation
3. Class weights experimentation vs custom scorer
4. Threshold tuning for cost-sensitive decisions
5. Visualization of expensive mistakes
6. Hyperparameter tuning with cost-aware scoring
7. Deep learning models with custom loss (if TensorFlow available)
8. Model comparison and results

## Usage

### Basic Usage

```python
from src.cost_sensitive_evaluation import (
    COST_MATRIX, compute_cost_score, compute_expected_cost
)
from src.cost_aware_models import train_all_cost_aware_models

# Train models with cost-aware evaluation
results = train_all_cost_aware_models(X_train, y_train, X_test, y_test)

# Access results
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Cost: {result['cost']:.4f}")
    print(f"  Expected Cost: {result['expected_cost']:.4f}")
```

### Threshold Tuning

```python
from src.cost_sensitive_evaluation import find_optimal_threshold

# Find optimal threshold for cost-sensitive decisions
threshold_results = find_optimal_threshold(y_test, y_proba)

print(f"Optimal Threshold: {threshold_results['optimal_threshold']:.2f}")
print(f"Optimal Cost: {threshold_results['optimal_cost']:.2f}")
```

### Analyzing Expensive Mistakes

```python
from src.cost_sensitive_evaluation import analyze_expensive_mistakes, print_cost_analysis

# Analyze expensive mistakes (negative→positive, positive→negative)
mistake_analysis = analyze_expensive_mistakes(
    y_test, y_pred, texts=tweet_texts, n_examples=20
)

# Print analysis
print_cost_analysis(mistake_analysis)
```

### Deep Learning with Custom Loss

```python
from src.deep_learning_models import train_feedforward_nn_on_embeddings

# Train feed-forward NN with custom cost-aware loss
result = train_feedforward_nn_on_embeddings(
    X_train_embeddings, y_train, X_test_embeddings, y_test,
    use_custom_loss=True  # Use cost-aware loss instead of cross-entropy
)

print(f"Cost: {result['cost']:.2f}")
print(f"Expected Cost: {result['expected_cost']:.2f}")
```

### Running the Complete Training Pipeline

```bash
cd stock_sentiment_analysis
python train_cost_sensitive.py
```

This will:
- Load and preprocess data
- Compute sentiment features and embeddings
- Train all cost-aware models
- Perform threshold tuning
- Analyze expensive mistakes
- Generate visualizations
- Save comparison results

## Model Comparison

Models are compared using:

1. **Accuracy**: Standard classification accuracy
2. **Cost**: Total cost based on cost matrix (lower is better)
3. **Expected Cost**: Expected cost using probability predictions (lower is better)
4. **F1-Macro**: Macro-averaged F1 score

The best model is selected based on **lowest cost**, not just accuracy.

## Key Features

### 1. Custom Loss for Deep Learning

The `custom_sentiment_loss()` function uses the cost matrix directly in the loss calculation, training the model to minimize expected asymmetric cost instead of plain cross-entropy.

### 2. Threshold Tuning

Models can be tuned by adjusting decision thresholds. For example, requiring higher confidence (e.g., p(positive) > 0.7) before predicting "positive" can reduce costly mistakes.

### 3. Class Weights

Experimenting with class weights (e.g., heavier weights for negative/positive vs neutral) can help models avoid expensive mistakes.

### 4. Cost-Aware Hyperparameter Tuning

Hyperparameter tuning uses cost-based scoring instead of accuracy, ensuring optimized models minimize expected cost.

## Visualization

The system generates several visualizations:

1. **Cost Matrix Heatmap**: Shows the cost structure
2. **Cost-Aware Confusion Matrix**: Annotated with cost information for each cell
3. **Model Comparison Table**: CSV file with all model metrics

All visualizations are saved to the `results/` directory.

## Results

After training, check:

- `results/cost_matrix.png` - Cost matrix visualization
- `results/cost_aware_confusion_matrix.png` - Annotated confusion matrix
- `results/cost_sensitive_model_comparison.csv` - Model comparison table

## Notes

- LightGBM is optional (will skip if not installed)
- TensorFlow is optional for deep learning models (will skip if not installed)
- The system handles both string labels ('negative', 'neutral', 'positive') and integer labels (0, 1, 2)
- All models are evaluated using both standard metrics and cost metrics

