# Advanced Model Training Guide

This guide covers the comprehensive model training pipeline with hyperparameter tuning, ensemble methods, and advanced evaluation.

## Quick Start

### 1. Data Verification

First, verify your data:

```bash
python verify_data.py
```

This will:
- Load `tweets_with_sentiment.csv`
- Display 20 random samples
- Show 3 samples per sentiment label
- Verify 50+ samples per sentiment class

### 2. Interactive Viewer

View your data in a web interface:

```bash
streamlit run tweet_viewer.py
```

Or use the simple HTML viewer:
- Open `simple_tweet_viewer.html` in your browser
- Load `results/tweets_with_sentiment.csv`

### 3. Advanced Model Training

Run the comprehensive training script:

```bash
python train_advanced_models.py
```

This will:
- Train baseline models (Logistic Regression, Random Forest, XGBoost)
- Run 5-fold TimeSeriesSplit cross-validation
- Perform hyperparameter tuning on top 2 models
- Create ensemble methods (Voting & Stacking)
- Generate SHAP explanations
- Save the best model

## What Gets Created

### Models
- `results/best_pipeline.pkl` - Best model pipeline (includes preprocessing)
- `results/best_model.pkl` - Best model only
- `results/w2v_model.pkl` - Word2Vec model
- `results/tfidf_vectorizer.pkl` - TF-IDF vectorizer

### Results
- `results/model_comparison.csv` - Performance comparison table
- `results/misclassified_examples.csv` - 20 misclassified tweet examples
- `results/error_pattern_analysis.txt` - Error pattern analysis
- `results/error_findings.md` - Documented error findings
- `results/model_metadata.json` - Model metadata

### Visualizations
- `results/comprehensive_evaluation.png` - All models comparison
- `results/error_analysis.png` - Error analysis plots
- `results/sentiment_distributions.png` - Train vs test distributions
- `results/shap_summary.png` - SHAP feature importance

## Model Pipeline Features

### ✅ Leakage Prevention
- **TimeSeriesSplit**: Training always precedes test in CV
- **Pipeline**: All preprocessing wrapped safely
- **ColumnTransformer**: Safe feature transformations
- **Embeddings**: Trained only on training data

### ✅ Evaluation
- **5-fold TimeSeriesSplit CV**: Mean ± std metrics
- **Test Set Evaluation**: Classification reports, ROC curves, confusion matrices
- **Comprehensive Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1-Score

### ✅ Advanced Features
- **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV
- **Ensemble Methods**: Voting Classifier & Stacking
- **SHAP Explanations**: Feature importance analysis
- **Error Pattern Analysis**: Systematic failure identification

## Using the Saved Model

### Load and Predict

```python
from predict import load_model, predict_from_merged_data
import pandas as pd

# Load model
pipeline = load_model()

# Load your data (must have all features computed)
merged_data = pd.read_csv('your_data.csv')

# Predict
results = predict_from_merged_data(merged_data, pipeline)
print(results[['tweet', 'predicted', 'prediction_probability']])
```

### Command Line

```bash
python predict.py --data results/merged_clean_with_embeddings.csv --output predictions.csv
```

## Model Comparison

The script automatically:
1. Trains 3 baseline models
2. Tunes top 2 models
3. Creates 2 ensemble methods
4. Compares all models
5. Selects best by test ROC-AUC

Expected improvement:
- **Baseline → Tuned**: 1-3% ROC-AUC improvement
- **Tuned → Ensemble**: 1-3% ROC-AUC improvement
- **Total**: 2-6% ROC-AUC improvement over baseline

## Error Analysis

The script identifies:
- **Finance Jargon Issues**: Neutral tweets with $ symbols misclassified
- **Sentiment Patterns**: Which sentiment classes have most errors
- **Target Patterns**: Errors by actual target value
- **Stock Patterns**: Which stocks have most errors

Findings are documented in `results/error_findings.md`.

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- scikit-learn (pipelines, CV, ensembles)
- xgboost (gradient boosting)
- shap (explanations)
- streamlit (interactive viewer)
- plotly (interactive charts)

## Troubleshooting

### SHAP not working?
- Install: `pip install shap`
- Some ensemble models may not support SHAP directly

### Out of memory?
- Reduce `n_samples` in SHAP explanations
- Use RandomizedSearchCV instead of GridSearchCV
- Reduce CV folds

### Model not found?
- Run `train_advanced_models.py` first
- Check `results/` directory for saved models

