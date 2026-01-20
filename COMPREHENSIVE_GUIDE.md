# Comprehensive Model Training & Evaluation Guide

This guide covers the complete implementation of advanced model training, evaluation, and deployment.

## 📋 Table of Contents

1. [Data Verification](#data-verification)
2. [Interactive Viewer](#interactive-viewer)
3. [Model Pipelines](#model-pipelines)
4. [Evaluation & Visualization](#evaluation--visualization)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Ensemble Methods](#ensemble-methods)
7. [Error Analysis](#error-analysis)
8. [Model Deployment](#model-deployment)

## 🔍 Data Verification

### Verify Data Structure

```bash
python verify_data.py
```

**What it does:**
- Loads `results/tweets_with_sentiment.csv`
- Displays 20 random samples: `df[['tweet', 'textblob_sentiment', 'vader_sentiment']].sample(20)`
- Groups by sentiment: `df.groupby('textblob_sentiment').head(3)`
- Verifies 50+ samples per sentiment class (positive/neutral/negative)

**Expected Output:**
```
✅ Loaded 63,676 rows
✅ All required columns present
✅ POSITIVE: 28,277 samples (≥50)
✅ NEUTRAL: 26,203 samples (≥50)
✅ NEGATIVE: 9,196 samples (≥50)
```

## 🖥️ Interactive Viewer

### Streamlit Viewer (Recommended)

```bash
streamlit run tweet_viewer.py
```

**Features:**
- ✅ Filter by stock, sentiment (TextBlob/VADER), target, date range
- ✅ Search tweets by keyword
- ✅ Color-coded sentiment labels in table view
- ✅ Interactive charts (sentiment distribution, trends over time)
- ✅ Statistics and agreement analysis
- ✅ Download filtered data as CSV

### Simple HTML Viewer (No Installation)

1. Open `simple_tweet_viewer.html` in your browser
2. Click "Select tweets_with_sentiment.csv file"
3. Browse and filter tweets

**Features:**
- No Python dependencies
- Works offline
- Fast and lightweight

## 🔧 Model Pipelines

### Leakage-Proof Implementation

All models use **scikit-learn Pipeline + ColumnTransformer**:

```python
from src.model_pipelines import (
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    create_xgboost_pipeline
)

# All preprocessing wrapped safely
lr_pipeline = create_logistic_regression_pipeline(use_scaling=True)
rf_pipeline = create_random_forest_pipeline(use_scaling=False)
xgb_pipeline = create_xgboost_pipeline(use_scaling=False)
```

**Key Features:**
- ✅ All transformations inside Pipeline
- ✅ Fit only on training folds during CV
- ✅ Transform test set separately
- ✅ No data leakage

### Time Series Splitting

```python
from src.model_pipelines import create_time_series_split

# Test set always from later period (no lookahead bias)
train_indices, test_indices = create_time_series_split(merged, test_size=0.2)
```

**TimeSeriesSplit for CV:**
- ✅ Training always precedes test in each fold
- ✅ 5-fold CV with chronological ordering
- ✅ All preprocessing inside CV loop

## 📊 Evaluation & Visualization

### Comprehensive CV Evaluation

```python
from src.advanced_evaluation import run_comprehensive_cv

cv_summary, cv_full = run_comprehensive_cv(
    pipeline, X_train, y_train, cv_folds=5
)

# Returns mean ± std for:
# - Accuracy
# - ROC-AUC
# - Precision
# - Recall
# - F1-Score
```

### Test Set Evaluation

**Classification Reports:**
- Precision, Recall, F1-Score per class
- Confusion matrices
- ROC curves for all models

**Visualizations Created:**
- `comprehensive_evaluation.png` - All models comparison
- `error_analysis.png` - Misclassification patterns
- `sentiment_distributions.png` - Train vs test distributions
- `model_evaluation.png` - ROC curves, confusion matrices

### Model Comparison Table

Automatically generated CSV with:
- Test Accuracy, ROC-AUC, Precision, Recall, F1
- CV ROC-AUC Mean ± Std
- Sorted by Test ROC-AUC

## 🎯 Hyperparameter Tuning

### Implementation

```python
from src.advanced_evaluation import hyperparameter_tuning

# Tune top 2 models
best_pipeline, best_params, best_cv_score = hyperparameter_tuning(
    pipeline, param_grid, X_train, y_train,
    cv_folds=5, method='random', n_iter=20
)
```

**Parameter Grids:**
- **Logistic Regression**: C, max_iter
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: n_estimators, max_depth, learning_rate

**Methods:**
- `GridSearchCV` - Exhaustive search (slower, thorough)
- `RandomizedSearchCV` - Random search (faster, good coverage)

**Expected Improvement:** 1-3% ROC-AUC over baseline

## 🎭 Ensemble Methods

### Voting Classifier

```python
from src.advanced_evaluation import create_ensemble_voting

ensemble = create_ensemble_voting(
    {'lr': lr_pipeline, 'rf': rf_pipeline, 'xgb': xgb_pipeline},
    voting='soft'  # or 'hard'
)
```

### Stacking Classifier

```python
from src.advanced_evaluation import create_ensemble_stacking

ensemble = create_ensemble_stacking(
    [('lr', lr_pipeline), ('rf', rf_pipeline), ('xgb', xgb_pipeline)],
    meta_estimator=LogisticRegression()  # Meta-learner
)
```

**Expected Improvement:** 1-3% ROC-AUC over best single model

## 🔬 Error Analysis

### Misclassification Analysis

**20 Misclassified Examples:**
- Saved to `misclassified_examples.csv`
- Includes tweet text, sentiment labels, true/predicted targets

**Error Patterns Identified:**
- By sentiment (positive/neutral/negative)
- By target (Up/Down)
- By stock
- Finance jargon issues ($ symbols)

### SHAP Explanations

```python
from src.advanced_evaluation import generate_shap_explanations

shap_values, explainer = generate_shap_explanations(
    model, X_test, n_samples=100
)
```

**Output:**
- `shap_summary.png` - Feature importance plot
- Identifies which features drive predictions

### Documented Findings

**Error Findings Document:**
- `error_findings.md` - Structured error analysis
- Identifies systematic failures
- Provides recommendations

**Example Finding:**
> **Issue:** Finance Jargon Confusion  
> **Description:** Model confuses neutral finance tweets as positive due to $ symbols  
> **Recommendation:** Consider feature engineering to handle finance symbols separately

## 💾 Model Deployment

### Save Best Model

The training script automatically saves:
- `best_pipeline.pkl` - Complete pipeline (preprocessing + model)
- `best_model.pkl` - Model only
- `model_metadata.json` - Performance metrics and metadata

### Load and Predict

```python
from predict import load_model, predict_from_merged_data

# Load model
pipeline = load_model()

# Predict on new data
results = predict_from_merged_data(merged_data, pipeline)
```

### Command Line

```bash
python predict.py --data your_data.csv --output predictions.csv
```

## 📈 Complete Workflow

### Step-by-Step

1. **Verify Data**
   ```bash
   python verify_data.py
   ```

2. **View Data** (Optional)
   ```bash
   streamlit run tweet_viewer.py
   ```

3. **Train Models**
   ```bash
   python train_advanced_models.py
   ```

4. **Review Results**
   - Check `results/model_comparison.csv`
   - Review `results/error_findings.md`
   - View visualizations in `results/`

5. **Use Model**
   ```python
   from predict import load_model
   pipeline = load_model()
   # Make predictions...
   ```

## 📊 Expected Results

### Model Performance

**Baseline Models:**
- Logistic Regression: ~0.55-0.65 ROC-AUC
- Random Forest: ~0.60-0.70 ROC-AUC
- XGBoost: ~0.62-0.72 ROC-AUC

**After Tuning:**
- +1-3% ROC-AUC improvement

**After Ensemble:**
- +1-3% additional improvement
- **Total: 2-6% improvement over baseline**

### Files Generated

**Models:**
- `best_pipeline.pkl`
- `best_model.pkl`
- `w2v_model.pkl`
- `tfidf_vectorizer.pkl`

**Results:**
- `model_comparison.csv`
- `misclassified_examples.csv`
- `error_pattern_analysis.txt`
- `error_findings.md`
- `model_metadata.json`

**Visualizations:**
- `comprehensive_evaluation.png`
- `error_analysis.png`
- `sentiment_distributions.png`
- `shap_summary.png`

## ✅ Verification Checklist

- [x] Data verification script created
- [x] Interactive viewer with filters and color-coding
- [x] Leakage-proof pipelines (Pipeline + ColumnTransformer)
- [x] TimeSeriesSplit for CV (5-fold)
- [x] Comprehensive evaluation (mean ± std metrics)
- [x] Model comparison table
- [x] Hyperparameter tuning (top 2 models)
- [x] Ensemble methods (Voting + Stacking)
- [x] Error pattern analysis
- [x] SHAP explanations
- [x] Model saving with prediction function
- [x] 20 misclassified examples
- [x] Train vs test distribution plots

## 🚀 Quick Start Commands

```bash
# 1. Verify data
python verify_data.py

# 2. View data (optional)
streamlit run tweet_viewer.py

# 3. Train all models
python train_advanced_models.py

# 4. Use saved model
python predict.py --data your_data.csv
```

All results will be saved in the `results/` directory!

