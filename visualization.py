"""
Visualization module for EDA and model evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .config import PLOT_STYLE, FIG_SIZE, RESULTS_DIR

# Set style
sns.set_style(PLOT_STYLE)
plt.rcParams['figure.figsize'] = FIG_SIZE
# Use non-interactive backend to prevent blocking
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_basic_eda(tweets, prices):
    """Plot basic exploratory data analysis visualizations."""
    print("Creating basic EDA plots...")
    with tqdm(total=4, desc="  Plotting") as pbar:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        pbar.update(1)
        
        # Handle both original and normalized column names
        stock_col = 'stock name' if 'stock name' in tweets.columns else 'Stock Name'
        date_col = 'date' if 'date' in tweets.columns else 'Date'
        close_col = 'close' if 'close' in prices.columns else 'Close'
        volume_col = 'volume' if 'volume' in prices.columns else 'Volume'
        
        # Tweet counts per stock
        tweet_counts = tweets[stock_col].value_counts()
        axes[0, 0].bar(tweet_counts.index, tweet_counts.values, color='steelblue')
        axes[0, 0].set_title('Tweet Count per Stock', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Stock')
        axes[0, 0].set_ylabel('Number of Tweets')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(tweet_counts.values):
            axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
        pbar.update(1)
        
        # Price statistics
        price_stats = prices.groupby(stock_col)[close_col].agg(['mean', 'std', 'min', 'max'])
        axes[0, 1].bar(price_stats.index, price_stats['mean'], yerr=price_stats['std'], 
                       capsize=5, color='coral', alpha=0.7)
        axes[0, 1].set_title('Average Closing Price by Stock (±1 std)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Stock')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        pbar.update(1)
        
        # Volume distribution
        volume_by_stock = prices.groupby(stock_col)[volume_col].mean()
        axes[1, 0].bar(volume_by_stock.index, volume_by_stock.values / 1e6, color='green', alpha=0.7)
        axes[1, 0].set_title('Average Daily Volume by Stock', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Stock')
        axes[1, 0].set_ylabel('Volume (Millions)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Price range over time
        unique_stocks = prices[stock_col].unique()
        for stock in tqdm(unique_stocks, desc="  Plotting stock trends", leave=False):
            stock_data = prices[prices[stock_col] == stock].sort_values(date_col)
            axes[1, 1].plot(stock_data[date_col], stock_data[close_col], label=stock, alpha=0.7, linewidth=2)
        axes[1, 1].set_title('Stock Price Trends Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Closing Price ($)')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        pbar.update(1)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'basic_eda.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to prevent blocking
        pbar.update(1)
    
    print("✅ Basic EDA plots saved to results/basic_eda.png")


def plot_advanced_eda(tweets, prices):
    """Plot advanced time series and correlation analysis."""
    print("Creating advanced EDA plots...")
    with tqdm(total=4, desc="  Plotting") as pbar:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        pbar.update(1)
        
        # Handle column names
        date_col = 'date' if 'date' in tweets.columns else 'Date'
        stock_col = 'stock name' if 'stock name' in prices.columns else 'Stock Name'
        close_col = 'close' if 'close' in prices.columns else 'Close'
        volume_col = 'volume' if 'volume' in prices.columns else 'Volume'
        
        # Daily tweet volume
        daily_tweets = tweets.groupby(date_col).size().reset_index(name='count')
        axes[0, 0].plot(daily_tweets[date_col], daily_tweets['count'], color='steelblue', linewidth=1.5)
        axes[0, 0].set_title('Daily Tweet Volume Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Tweets')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        pbar.update(1)
        
        # Volatility
        prices_sorted = prices.sort_values([stock_col, date_col])
        prices_sorted['returns'] = prices_sorted.groupby(stock_col)[close_col].pct_change()
        prices_sorted['volatility'] = prices_sorted.groupby(stock_col)['returns'].rolling(
            window=30
        ).std().reset_index(0, drop=True)
        
        unique_stocks = prices_sorted[stock_col].unique()
        for stock in tqdm(unique_stocks, desc="  Plotting volatility", leave=False):
            stock_data = prices_sorted[prices_sorted[stock_col] == stock].dropna(subset=['volatility'])
            axes[0, 1].plot(stock_data[date_col], stock_data['volatility'], label=stock, alpha=0.7, linewidth=1.5)
        axes[0, 1].set_title('30-Day Rolling Volatility by Stock', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        pbar.update(1)
        
        # Returns distribution
        returns_data = prices_sorted['returns'].dropna()
        axes[1, 0].hist(returns_data, bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(returns_data.mean(), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {returns_data.mean():.4f}')
        axes[1, 0].set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volume vs Price
        for stock in tqdm(prices[stock_col].unique(), desc="  Plotting volume vs price", leave=False):
            stock_data = prices[prices[stock_col] == stock]
            axes[1, 1].scatter(stock_data[volume_col] / 1e6, stock_data[close_col], 
                              label=stock, alpha=0.5, s=20)
        axes[1, 1].set_title('Volume vs Closing Price', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Volume (Millions)')
        axes[1, 1].set_ylabel('Closing Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        pbar.update(1)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'advanced_eda.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to prevent blocking
        pbar.update(1)
    
    print("✅ Advanced EDA plots saved to results/advanced_eda.png")


def plot_correlation_matrix(merged):
    """Plot correlation heatmap of sentiment and price features."""
    corr_features = [
        'tb_polarity', 'tb_subjectivity', 'vader_compound',
        'tweet_length', 'word_count', 'hashtag_count',
        'count_buy', 'count_sell', 'count_bullish', 'count_bearish'
    ]
    
    # Add technical indicators if available
    tech_features = ['ma_5', 'ma_10', 'price_change_pct', 'rsi', 'volatility']
    corr_features.extend([f for f in tech_features if f in merged.columns])
    
    # Add price and target
    if 'close' in merged.columns:
        corr_features.append('close')
    if 'volume' in merged.columns:
        corr_features.append('volume')
    if 'next_day_change' in merged.columns:
        corr_features.append('next_day_change')
    
    # Filter to available columns
    corr_features = [f for f in corr_features if f in merged.columns]
    corr_matrix = merged[corr_features].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Sentiment & Price Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_analysis(merged):
    """Plot sentiment distribution and trends."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sentiment distribution (TextBlob)
    if 'tb_label' in merged.columns:
        sentiment_counts = merged['tb_label'].value_counts()
        # Order: positive, neutral, negative
        order = ['positive', 'neutral', 'negative']
        sentiment_counts = sentiment_counts.reindex([s for s in order if s in sentiment_counts.index])
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        bar_colors = [colors.get(s, 'gray') for s in sentiment_counts.index]
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                      color=bar_colors, alpha=0.7)
        axes[0, 0].set_title('TextBlob Sentiment Distribution\n(Positive, Neutral, Negative)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Sentiment Label')
        axes[0, 0].set_ylabel('Count')
        for i, (sent, v) in enumerate(sentiment_counts.items()):
            axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # Polarity distribution with labels
    if 'tb_polarity' in merged.columns:
        axes[0, 1].hist(merged['tb_polarity'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0.05, color='green', linestyle='--', linewidth=1, label='Positive threshold')
        axes[0, 1].axvline(-0.05, color='red', linestyle='--', linewidth=1, label='Negative threshold')
        axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[0, 1].set_title('TextBlob Polarity Distribution\n(>0.05=Positive, <-0.05=Negative, else=Neutral)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Polarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Sentiment vs Price Change
    if 'next_day_change' in merged.columns and 'tb_label' in merged.columns:
        sns.boxplot(data=merged, x='tb_label', y='next_day_change', ax=axes[1, 0],
                   order=['positive', 'neutral', 'negative'])
        axes[1, 0].set_title('Next-Day Price Change by Sentiment Label', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Sentiment Label (Positive, Neutral, Negative)')
        axes[1, 0].set_ylabel('Price Change ($)')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
    
    # VADER vs TextBlob comparison
    if 'vader_label' in merged.columns and 'tb_label' in merged.columns:
        # Create comparison matrix
        comparison = pd.crosstab(merged['tb_label'], merged['vader_label'], 
                                 normalize='index') * 100
        sns.heatmap(comparison, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1],
                   cbar_kws={'label': 'Percentage'})
        axes[1, 1].set_title('Sentiment Label Agreement\n(TextBlob vs VADER)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('VADER Sentiment')
        axes[1, 1].set_ylabel('TextBlob Sentiment')
    elif 'date' in merged.columns and 'tb_polarity' in merged.columns:
        # Fallback: sentiment over time
        daily_sentiment = merged.groupby('date')['tb_polarity'].mean().reset_index()
        axes[1, 1].plot(daily_sentiment['date'], daily_sentiment['tb_polarity'], 
                        color='purple', linewidth=2, alpha=0.8)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Average Daily Sentiment Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Average Polarity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_evaluation(results, best_model_name, y_test, y_pred_best=None, y_proba_best=None):
    """
    Plot comprehensive model evaluation metrics for all models.
    
    Args:
        results: Dictionary with results for all models
        best_model_name: Name of the best model
        y_test: True labels
        y_pred_best: Predictions from best model (optional, will use from results if None)
        y_proba_best: Probabilities from best model (optional, will use from results if None)
    """
    from sklearn.metrics import confusion_matrix, roc_curve
    
    if y_pred_best is None:
        y_pred_best = results[best_model_name]['predictions']
    if y_proba_best is None:
        y_proba_best = results[best_model_name]['probabilities']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Confusion Matrix for Best Model
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], cbar_kws={'shrink': 0.8})
    axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curves for All Models
    colors = {'Logistic Regression': 'steelblue', 'Random Forest': 'green', 'XGBoost': 'coral'}
    for model_name, model_results in results.items():
        fpr, tpr, _ = roc_curve(y_test, model_results['probabilities'])
        axes[0, 1].plot(fpr, tpr, linewidth=2, color=colors.get(model_name, 'gray'),
                       label=f'{model_name} (AUC = {model_results["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves - All Models', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Probability Distribution for Best Model
    axes[0, 2].hist(y_proba_best, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[0, 2].set_title(f'Prediction Probability Distribution - {best_model_name}', 
                        fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Predicted Probability')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Model Comparison - Test Set Performance
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    roc_aucs = [results[m]['roc_auc'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8, color='coral')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Model Performance Comparison (Test Set)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # 5. Cross-Validation vs Test Set Comparison
    cv_accuracies = [results[m].get('cv_accuracy_mean', 0) for m in model_names]
    cv_roc_aucs = [results[m].get('cv_roc_auc_mean', 0) for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.2
    axes[1, 1].bar(x - 1.5*width, accuracies, width, label='Test Accuracy', alpha=0.8, color='steelblue')
    axes[1, 1].bar(x - 0.5*width, cv_accuracies, width, label='CV Accuracy', alpha=0.8, color='lightblue')
    axes[1, 1].bar(x + 0.5*width, roc_aucs, width, label='Test ROC-AUC', alpha=0.8, color='coral')
    axes[1, 1].bar(x + 1.5*width, cv_roc_aucs, width, label='CV ROC-AUC', alpha=0.8, color='lightcoral')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('CV vs Test Set Performance', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    
    # 6. Sentiment Distribution by Prediction (for best model)
    # This will be added if merged data is available
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Sentiment analysis\navailable in separate plot', 
                   ha='center', va='center', fontsize=12, style='italic')
    axes[1, 2].set_title('Sentiment Analysis', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(best_model, X_features, best_model_name, top_n=20, pipeline=None):
    """
    Plot feature importance from the best model.
    
    Args:
        best_model: Trained model with feature_importances_ or coef_ attribute
        X_features: Feature matrix (for column names) - should be the original features before pipeline
        best_model_name: Name of the model
        top_n: Number of top features to display
        pipeline: Optional pipeline to extract feature names after transformations
    
    Returns:
        Series: Top features with their importances
    """
    # Get feature names - handle both DataFrame and array cases
    if hasattr(X_features, 'columns'):
        feature_names = X_features.columns.tolist()
    else:
        # If array, create generic names
        feature_names = [f'feature_{i}' for i in range(X_features.shape[1])]
    
    # If pipeline has feature selection, get selected feature names
    if pipeline is not None and 'feature_selection' in pipeline.named_steps:
        selector = pipeline.named_steps['feature_selection']
        if hasattr(selector, 'selected_features_'):
            # CVFeatureSelector stores feature names
            if isinstance(selector.selected_features_, list):
                feature_names = selector.selected_features_
            elif hasattr(selector, 'get_support'):
                # Standard sklearn selector
                selected_mask = selector.get_support()
                feature_names = [f for f, m in zip(feature_names, selected_mask) if m]
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost)
        if len(feature_names) != len(best_model.feature_importances_):
            # Feature selection was applied, use only selected features
            importances = pd.Series(best_model.feature_importances_, 
                                  index=feature_names[:len(best_model.feature_importances_)])
        else:
            importances = pd.Series(best_model.feature_importances_, index=feature_names)
        top_features = importances.sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh', color='steelblue')
        plt.title(f'Top {top_n} Feature Importances - {best_model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_features
    elif hasattr(best_model, 'coef_'):
        # For Logistic Regression
        coef = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
        if len(feature_names) != len(coef):
            importances = pd.Series(np.abs(coef), 
                                  index=feature_names[:len(coef)])
        else:
            importances = pd.Series(np.abs(coef), index=feature_names)
        top_features = importances.sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh', color='steelblue')
        plt.title(f'Top {top_n} Feature Coefficients (abs) - {best_model_name}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Coefficient Magnitude')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_features
    return None


def plot_error_analysis(results, best_model_name, y_test, merged_test=None):
    """
    Analyze and visualize model errors (misclassifications).
    
    Args:
        results: Dictionary with model results
        best_model_name: Name of the best model
        y_test: True labels
        merged_test: Test data with original features (optional, for detailed analysis)
    """
    from sklearn.metrics import confusion_matrix
    
    y_pred = results[best_model_name]['predictions']
    y_proba = results[best_model_name]['probabilities']
    
    # Identify misclassifications
    misclassified = y_test != y_pred
    false_positives = (y_test == 0) & (y_pred == 1)
    false_negatives = (y_test == 1) & (y_pred == 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Misclassification by probability
    axes[0, 0].hist(y_proba[~misclassified], bins=30, alpha=0.6, label='Correct', color='green')
    axes[0, 0].hist(y_proba[misclassified], bins=30, alpha=0.6, label='Misclassified', color='red')
    axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Prediction Probability: Correct vs Misclassified', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. False Positives vs False Negatives
    fp_count = false_positives.sum()
    fn_count = false_negatives.sum()
    axes[0, 1].bar(['False Positives', 'False Negatives'], [fp_count, fn_count], 
                   color=['red', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Error Types Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([fp_count, fn_count]):
        axes[0, 1].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Probability distribution by error type
    if fp_count > 0 and fn_count > 0:
        axes[1, 0].hist(y_proba[false_positives], bins=20, alpha=0.6, 
                       label=f'False Positives (n={fp_count})', color='red')
        axes[1, 0].hist(y_proba[false_negatives], bins=20, alpha=0.6, 
                       label=f'False Negatives (n={fn_count})', color='orange')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Probability Distribution by Error Type', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds', ax=axes[1, 1],
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print error statistics
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total misclassifications: {misclassified.sum()} ({misclassified.mean()*100:.2f}%)")
    print(f"False Positives (predicted Up, actual Down): {fp_count}")
    print(f"False Negatives (predicted Down, actual Up): {fn_count}")
    print(f"True Positives: {(y_test == 1) & (y_pred == 1)}")
    print(f"True Negatives: {(y_test == 0) & (y_pred == 0)}")


def plot_sentiment_distributions(merged_train, merged_test=None, y_test=None, y_pred=None):
    """
    Visualize sentiment and target distributions.
    Uses training set for exploratory analysis, test set for honest assessment.
    
    Args:
        merged_train: Training data with sentiment features
        merged_test: Test data (optional)
        y_test: Test labels (optional)
        y_pred: Test predictions (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sentiment distribution in training set
    if 'tb_label' in merged_train.columns:
        sentiment_counts = merged_train['tb_label'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                      color=['green', 'red', 'gray'], alpha=0.7)
        axes[0, 0].set_title('Sentiment Distribution (Training Set)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(sentiment_counts.values):
            axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Target distribution in training set
    if 'target' in merged_train.columns:
        target_counts = merged_train['target'].value_counts()
        axes[0, 1].bar(['Down', 'Up'], [target_counts.get(0, 0), target_counts.get(1, 0)], 
                      color=['red', 'green'], alpha=0.7)
        axes[0, 1].set_title('Target Distribution (Training Set)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Target')
        axes[0, 1].set_ylabel('Count')
        for i, v in enumerate([target_counts.get(0, 0), target_counts.get(1, 0)]):
            axes[0, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Sentiment vs Target in training set
    if 'tb_polarity' in merged_train.columns and 'target' in merged_train.columns:
        sns.boxplot(data=merged_train, x='target', y='tb_polarity', ax=axes[1, 0])
        axes[1, 0].set_xticklabels(['Down', 'Up'])
        axes[1, 0].set_title('Sentiment Polarity by Target (Training Set)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Target')
        axes[1, 0].set_ylabel('Polarity Score')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Test set comparison (if available)
    if merged_test is not None and y_test is not None and y_pred is not None:
        # Compare actual vs predicted distribution
        actual_dist = pd.Series(y_test).value_counts().sort_index()
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        x = np.arange(len(actual_dist))
        width = 0.35
        axes[1, 1].bar(x - width/2, [actual_dist.get(0, 0), actual_dist.get(1, 0)], 
                      width, label='Actual', alpha=0.8, color='steelblue')
        axes[1, 1].bar(x + width/2, [pred_dist.get(0, 0), pred_dist.get(1, 0)], 
                      width, label='Predicted', alpha=0.8, color='coral')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Down', 'Up'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Target Distribution: Actual vs Predicted (Test Set)', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Test set analysis\nrequires predictions', 
                       ha='center', va='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'sentiment_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(predictions_df, true_labels=None, date_col=None, stock_col=None):
    """
    Create comprehensive visualizations for model predictions.
    
    Args:
        predictions_df: DataFrame with predictions (must have 'predicted' and 'prediction_probability' columns)
        true_labels: True labels (optional, for evaluation metrics)
        date_col: Name of date column (optional, for time series plots)
        stock_col: Name of stock column (optional, for stock-specific plots)
    """
    print("Creating prediction visualizations...")
    
    # Check required columns
    if 'predicted' not in predictions_df.columns:
        raise ValueError("predictions_df must have 'predicted' column")
    if 'prediction_probability' not in predictions_df.columns:
        raise ValueError("predictions_df must have 'prediction_probability' column")
    
    # Determine date and stock columns
    if date_col is None:
        date_col = 'date' if 'date' in predictions_df.columns else ('Date' if 'Date' in predictions_df.columns else None)
    if stock_col is None:
        stock_col = 'stock name' if 'stock name' in predictions_df.columns else ('Stock Name' if 'Stock Name' in predictions_df.columns else None)
    
    # Create figure with subplots
    n_plots = 6
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    with tqdm(total=n_plots, desc="  Creating plots") as pbar:
        # 1. Prediction Distribution (Up vs Down)
        ax1 = fig.add_subplot(gs[0, 0])
        pred_counts = predictions_df['predicted'].value_counts().sort_index()
        colors = ['red' if x == 0 else 'green' for x in pred_counts.index]
        bars = ax1.bar(['Down', 'Up'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (bar, count) in enumerate(zip(bars, [pred_counts.get(0, 0), pred_counts.get(1, 0)])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        pbar.update(1)
        
        # 2. Prediction Probability Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(predictions_df['prediction_probability'], bins=50, color='steelblue', 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        pbar.update(1)
        
        # 3. Prediction Confidence (distance from 0.5)
        ax3 = fig.add_subplot(gs[0, 2])
        confidence = np.abs(predictions_df['prediction_probability'] - 0.5) * 2  # Scale to 0-1
        ax3.hist(confidence, bins=30, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Confidence (Distance from 0.5)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        mean_conf = confidence.mean()
        ax3.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_conf:.3f}')
        ax3.legend()
        pbar.update(1)
        
        # 4. Predictions Over Time (if date column exists)
        ax4 = fig.add_subplot(gs[1, :2])
        if date_col and date_col in predictions_df.columns:
            # Convert date to datetime if needed
            pred_with_date = predictions_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(pred_with_date[date_col]):
                pred_with_date[date_col] = pd.to_datetime(pred_with_date[date_col], errors='coerce')
            
            # Group by date and calculate daily prediction rates
            daily_preds = pred_with_date.groupby(pred_with_date[date_col].dt.date).agg({
                'predicted': ['count', 'sum'],
                'prediction_probability': 'mean'
            }).reset_index()
            daily_preds.columns = ['date', 'total', 'up_count', 'avg_probability']
            daily_preds['up_rate'] = daily_preds['up_count'] / daily_preds['total']
            daily_preds['date'] = pd.to_datetime(daily_preds['date'])
            daily_preds = daily_preds.sort_values('date')
            
            # Plot up rate over time
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(daily_preds['date'], daily_preds['up_rate'], 
                           color='green', linewidth=2, label='Up Prediction Rate', marker='o', markersize=3)
            line2 = ax4_twin.plot(daily_preds['date'], daily_preds['avg_probability'], 
                                color='blue', linewidth=2, label='Avg Probability', 
                                linestyle='--', alpha=0.7)
            ax4.axhline(0.5, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Up Prediction Rate', color='green')
            ax4_twin.set_ylabel('Average Probability', color='blue')
            ax4.set_title('Predictions Over Time', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
        else:
            ax4.text(0.5, 0.5, 'Date column not available\nfor time series plot', 
                     ha='center', va='center', fontsize=12, style='italic', transform=ax4.transAxes)
            ax4.set_title('Predictions Over Time (No Date Column)', fontsize=12, fontweight='bold')
        pbar.update(1)
        
        # 5. Predictions by Stock (if stock column exists)
        ax5 = fig.add_subplot(gs[1, 2])
        if stock_col and stock_col in predictions_df.columns:
            stock_preds = predictions_df.groupby(stock_col)['predicted'].agg(['count', 'sum']).reset_index()
            stock_preds.columns = ['stock', 'total', 'up_count']
            stock_preds['up_rate'] = stock_preds['up_count'] / stock_preds['total']
            stock_preds = stock_preds.sort_values('up_rate', ascending=False)
            
            bars = ax5.barh(stock_preds['stock'], stock_preds['up_rate'], 
                          color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
            ax5.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax5.set_xlabel('Up Prediction Rate')
            ax5.set_title('Predictions by Stock', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, stock_preds['total'])):
                ax5.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        f' (n={count})', va='center', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Stock column not available', 
                     ha='center', va='center', fontsize=12, style='italic', transform=ax5.transAxes)
            ax5.set_title('Predictions by Stock (No Stock Column)', fontsize=12, fontweight='bold')
        pbar.update(1)
        
        # 6. Confusion Matrix & Metrics (if true labels provided)
        ax6 = fig.add_subplot(gs[2, :])
        if true_labels is not None:
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
            
            y_true = true_labels
            y_pred = predictions_df['predicted'].values
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create confusion matrix heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                       cbar_kws={'label': 'Count'}, square=True, linewidths=1)
            ax6.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax6.set_ylabel('Actual', fontsize=11, fontweight='bold')
            ax6.set_title(f'Confusion Matrix (Accuracy: {accuracy:.3f})', 
                         fontsize=12, fontweight='bold', pad=20)
            
            # Add classification report as text
            report = classification_report(y_true, y_pred, target_names=['Down', 'Up'], 
                                          output_dict=True, zero_division=0)
            report_text = f"Accuracy: {accuracy:.3f}\n"
            report_text += f"Precision (Up): {report['1']['precision']:.3f}\n"
            report_text += f"Recall (Up): {report['1']['recall']:.3f}\n"
            report_text += f"F1-Score (Up): {report['1']['f1-score']:.3f}"
            ax6.text(1.15, 0.5, report_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, 'True labels not provided\nCannot compute accuracy metrics', 
                     ha='center', va='center', fontsize=12, style='italic', transform=ax6.transAxes)
            ax6.set_title('Confusion Matrix (True Labels Not Provided)', 
                         fontsize=12, fontweight='bold')
        pbar.update(1)
    
    plt.suptitle('Model Predictions Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(RESULTS_DIR / 'predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Prediction visualizations saved to results/predictions_analysis.png")
    
    # Print summary statistics
    print("\n📊 Prediction Summary:")
    print(f"  Total predictions: {len(predictions_df):,}")
    up_count = (predictions_df['predicted'] == 1).sum()
    down_count = (predictions_df['predicted'] == 0).sum()
    print(f"  Up predictions: {up_count:,} ({up_count/len(predictions_df)*100:.1f}%)")
    print(f"  Down predictions: {down_count:,} ({down_count/len(predictions_df)*100:.1f}%)")
    print(f"  Mean probability: {predictions_df['prediction_probability'].mean():.3f}")
    print(f"  Std probability: {predictions_df['prediction_probability'].std():.3f}")
    print(f"  Min probability: {predictions_df['prediction_probability'].min():.3f}")
    print(f"  Max probability: {predictions_df['prediction_probability'].max():.3f}")
    
    if true_labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(true_labels, predictions_df['predicted'])
        precision = precision_score(true_labels, predictions_df['predicted'], zero_division=0)
        recall = recall_score(true_labels, predictions_df['predicted'], zero_division=0)
        f1 = f1_score(true_labels, predictions_df['predicted'], zero_division=0)
        print(f"\n📈 Performance Metrics:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")

