"""
Configuration file for Stock Sentiment Analysis project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PARENT_DIR = PROJECT_ROOT.parent

# Data file paths (assuming CSV files are in parent directory)
TWEETS_CSV = PARENT_DIR / "stock_tweets.csv"
PRICES_CSV = PARENT_DIR / "stock_yfinance_data.csv"

# Output file paths
MERGED_OUTPUT = RESULTS_DIR / "merged_clean_with_embeddings.csv"
PREDICTIONS_OUTPUT = RESULTS_DIR / "model_predictions.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15

# Gradient Boosting parameters
GB_N_ESTIMATORS = 200
GB_MAX_DEPTH = 5

# XGBoost parameters
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1

# Word2Vec parameters
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2

# TF-IDF parameters
TFIDF_MAX_FEATURES = 5000

# Feature selection parameters
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'mutual_info'  # Options: 'mutual_info', 'f_classif', 'rfe', 'select_from_model'
FEATURE_SELECTION_N_FEATURES = None  # None = auto (50% or max 500)
FEATURE_SELECTION_CV_FOLDS = 5

# Technical indicator parameters
MA_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
VOLATILITY_WINDOW = 30
VOLUME_MA_WINDOW = 10

# Sentiment thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Finance keywords
FINANCE_KEYWORDS = ['buy', 'sell', 'bullish', 'bearish', 'earnings', 'ipo']

# Visualization settings
PLOT_STYLE = "whitegrid"
FIG_SIZE = (12, 6)

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

