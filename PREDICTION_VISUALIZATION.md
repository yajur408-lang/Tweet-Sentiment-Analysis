# Prediction Visualization Guide

This guide explains how to create and view comprehensive visualizations for model predictions.

## 📊 What Gets Visualized

The prediction visualization function creates a comprehensive 6-panel dashboard showing:

1. **Prediction Distribution** - Bar chart showing Up vs Down predictions
2. **Probability Histogram** - Distribution of prediction probabilities
3. **Confidence Distribution** - How confident the model is in its predictions
4. **Predictions Over Time** - Time series showing prediction trends (if date column available)
5. **Predictions by Stock** - Stock-specific prediction rates (if stock column available)
6. **Confusion Matrix & Metrics** - Accuracy metrics (if true labels provided)

## 🚀 How to Use

### Method 1: Automatic (During Training)

When you run `train_advanced_models.py`, prediction visualizations are automatically generated:

```bash
python train_advanced_models.py
```

This will:
- Train all models
- Generate predictions on test set
- Automatically create `results/predictions_analysis.png`
- Save test predictions to `results/test_predictions.csv`

### Method 2: From Prediction Script

When making predictions with the `predict.py` script, use the `--plot` flag:

```bash
python predict.py --data your_data.csv --plot
```

For evaluation metrics, also provide true labels:

```bash
python predict.py --data your_data.csv --plot --true-labels target
```

### Method 3: Standalone Visualization Script

If you already have a CSV file with predictions, use the standalone visualization script:

```bash
python visualize_predictions.py --data predictions.csv
```

With true labels for evaluation:

```bash
python visualize_predictions.py --data predictions.csv --true-labels target
```

With custom column names:

```bash
python visualize_predictions.py --data predictions.csv \
    --true-labels target \
    --date-col date \
    --stock-col stock_name
```

## 📋 Required CSV Format

Your CSV file must have these columns:
- `predicted` - Binary predictions (0=Down, 1=Up)
- `prediction_probability` - Probability scores (0-1)

Optional columns (for enhanced visualizations):
- `target` or custom column name - True labels (for evaluation)
- `date` or `Date` - For time series plots
- `stock name` or `Stock Name` - For stock-specific analysis

## 📈 Example Output

After running, you'll get:

1. **Visualization File**: `results/predictions_analysis.png`
   - 6-panel comprehensive dashboard
   - High-resolution (300 DPI)
   - Ready for reports/presentations

2. **Console Output**:
   ```
   📊 Prediction Summary:
     Total predictions: 12,735
     Up predictions: 6,420 (50.4%)
     Down predictions: 6,315 (49.6%)
     Mean probability: 0.502
     Std probability: 0.187
     Min probability: 0.023
     Max probability: 0.987
   
   📈 Performance Metrics:
     Accuracy: 0.625
     Precision: 0.631
     Recall: 0.642
     F1-Score: 0.636
   ```

## 🎯 Use Cases

### 1. Model Evaluation
Visualize test set predictions to understand model performance:
```bash
python visualize_predictions.py --data results/test_predictions.csv --true-labels target
```

### 2. New Data Predictions
After making predictions on new data:
```bash
python predict.py --data new_data.csv --plot
```

### 3. Custom Analysis
Analyze predictions with specific columns:
```bash
python visualize_predictions.py --data predictions.csv \
    --date-col timestamp \
    --stock-col ticker \
    --true-labels actual_label
```

## 📁 Output Files

- `results/predictions_analysis.png` - Main visualization dashboard
- `results/test_predictions.csv` - Test set predictions (from training)
- `results/predictions.csv` - General predictions (from predict.py)

## 🔍 Understanding the Plots

### Prediction Distribution
Shows how many Up vs Down predictions the model made. Useful for checking class balance.

### Probability Histogram
Shows the distribution of prediction probabilities. A bimodal distribution (peaks near 0 and 1) indicates high confidence predictions.

### Confidence Distribution
Measures how far predictions are from the 0.5 threshold. Higher values = more confident predictions.

### Predictions Over Time
Time series showing:
- Up prediction rate (green line)
- Average probability (blue dashed line)
- Useful for identifying trends

### Predictions by Stock
Shows which stocks the model predicts as "Up" more often. Useful for identifying stock-specific patterns.

### Confusion Matrix
Only shown if true labels are provided. Shows:
- True Positives, False Positives
- True Negatives, False Negatives
- Accuracy, Precision, Recall, F1-Score

## 💡 Tips

1. **Always provide true labels** when available for comprehensive evaluation
2. **Include date column** for time series analysis
3. **Include stock column** for stock-specific insights
4. **Check probability distribution** - if most predictions are near 0.5, model is uncertain
5. **Compare train vs test** predictions to check for overfitting

## 🐛 Troubleshooting

**Error: "predicted column not found"**
- Make sure your CSV has a `predicted` column with 0/1 values

**Error: "prediction_probability column not found"**
- Make sure your CSV has a `prediction_probability` column with 0-1 values

**No time series plot shown**
- Check if your CSV has a `date` or `Date` column
- Use `--date-col` to specify custom column name

**No stock plot shown**
- Check if your CSV has a `stock name` or `Stock Name` column
- Use `--stock-col` to specify custom column name

## 📚 Related Files

- `src/visualization.py` - Contains `plot_predictions()` function
- `predict.py` - Prediction script with `--plot` option
- `visualize_predictions.py` - Standalone visualization script
- `train_advanced_models.py` - Training script (auto-generates visualizations)

