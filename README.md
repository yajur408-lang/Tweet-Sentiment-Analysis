# Stock Sentiment Analysis Project

A comprehensive NLP and machine learning project for predicting stock price movements using Twitter sentiment analysis.

## Project Structure

```
stock_sentiment_analysis/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── utils.py               # Utility functions
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── sentiment_analysis.py  # Sentiment computation (TextBlob, VADER)
│   ├── feature_engineering.py # Technical indicators
│   ├── embeddings.py          # Word2Vec embeddings
│   ├── models.py              # ML models (RF, GB, XGBoost)
│   └── visualization.py       # Plotting functions
├── data/                      # Data directory (optional)
├── results/                   # Output files and plots
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Sentiment Analysis**: TextBlob and VADER sentiment scoring
- **Feature Engineering**: Technical indicators (MA, RSI, Volatility) - all backward-looking
- **Word Embeddings**: Word2Vec and TF-IDF for tweet representation
- **Feature Selection**: Cross-validation based feature selection (prevents data leakage)
- **Multiple ML Models**: Random Forest, Gradient Boosting, XGBoost
- **Comprehensive EDA**: Time series analysis, correlation matrices
- **Model Evaluation**: ROC curves, confusion matrices, feature importance
- **Data Leakage Prevention**: Proper train/test splitting, CV-based feature selection

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
```

## Usage

### Run Complete Analysis

```bash
python main.py
```

This will:
1. Load and preprocess data
2. Perform exploratory data analysis
3. Compute sentiment features
4. Create technical indicators
5. Train Word2Vec embeddings
6. Train and compare multiple ML models
7. Generate visualizations
8. Save results to `results/` directory

### Data Requirements

Place your CSV files in the parent directory:
- `stock_tweets.csv` - Tweets data with columns: Date, Stock Name, Tweet
- `stock_yfinance_data.csv` - Stock price data with columns: Date, Stock Name, Open, High, Low, Close, Volume

### Output Files

- `results/merged_clean_with_embeddings.csv` - Processed dataset with all features
- `results/model_predictions.csv` - Model predictions on test set
- `results/*.png` - Visualization plots

## Configuration

Edit `src/config.py` to customize:
- Model parameters (n_estimators, max_depth, etc.)
- Feature engineering parameters (window sizes, etc.)
- Feature selection settings (method, number of features, CV folds)
- File paths
- Visualization settings

### Feature Selection Options

- `USE_FEATURE_SELECTION`: Enable/disable feature selection (default: True)
- `FEATURE_SELECTION_METHOD`: Method to use ('mutual_info', 'f_classif', 'rfe', 'select_from_model')
- `FEATURE_SELECTION_N_FEATURES`: Number of features to select (None = auto)
- `FEATURE_SELECTION_CV_FOLDS`: Number of CV folds for selection (default: 5)

## Module Usage

You can also import and use individual modules:

```python
from src.data_loader import load_data, preprocess_data
from src.sentiment_analysis import compute_all_sentiment_features
from src.models import train_all_models

# Load data
tweets, prices = load_data()
tweets, prices = preprocess_data(tweets, prices)

# Compute sentiment
merged = compute_all_sentiment_features(merged)

# Train models
results, best_model = train_all_models(X_train, X_test, y_train, y_test)
```

## Model Performance

The script automatically compares three models:
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting algorithm
- **XGBoost**: Optimized gradient boosting

The best model (by ROC-AUC) is selected and evaluated in detail.

## License

This project is for educational purposes.

