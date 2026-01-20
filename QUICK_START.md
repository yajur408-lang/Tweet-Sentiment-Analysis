# Quick Start Guide

## 🚀 Run Everything

### 1. Verify Your Data
```bash
python verify_data.py
```
Shows samples and verifies 50+ tweets per sentiment class.

### 2. View Data Interactively
```bash
streamlit run tweet_viewer.py
```
Or open `simple_tweet_viewer.html` in your browser.

### 3. Train Advanced Models
```bash
python train_advanced_models.py
```

This will:
- ✅ Train 3 baseline models (LR, RF, XGBoost)
- ✅ Run 5-fold TimeSeriesSplit CV
- ✅ Tune hyperparameters (top 2 models)
- ✅ Create ensembles (Voting + Stacking)
- ✅ Generate SHAP explanations
- ✅ Analyze errors
- ✅ Save best model

### 4. Use Saved Model
```python
from predict import load_model
pipeline = load_model()
# Make predictions...
```

## 📁 What Gets Created

**Models:**
- `results/best_pipeline.pkl` - Best model (use this!)
- `results/best_model.pkl` - Model only
- `results/w2v_model.pkl` - Word2Vec embeddings
- `results/tfidf_vectorizer.pkl` - TF-IDF vectorizer

**Results:**
- `results/model_comparison.csv` - Performance table
- `results/misclassified_examples.csv` - 20 error examples
- `results/error_findings.md` - Error analysis
- `results/model_metadata.json` - Model info

**Plots:**
- `results/comprehensive_evaluation.png`
- `results/error_analysis.png`
- `results/sentiment_distributions.png`
- `results/shap_summary.png`

## 🎯 Expected Performance

- **Baseline Models:** 0.55-0.72 ROC-AUC
- **After Tuning:** +1-3% improvement
- **After Ensemble:** +1-3% additional
- **Total:** 2-6% improvement over baseline

## ✅ All Requirements Met

- [x] Data verification with samples
- [x] Interactive viewer with filters
- [x] Leakage-proof pipelines
- [x] TimeSeriesSplit CV (5-fold)
- [x] Comprehensive evaluation
- [x] Model comparison table
- [x] Hyperparameter tuning
- [x] Ensemble methods
- [x] Error analysis & SHAP
- [x] Model saving & prediction function

