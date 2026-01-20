"""
Advanced error pattern analysis
Identifies systematic failures and error patterns
"""
import pandas as pd
import numpy as np
from .config import RESULTS_DIR


def analyze_error_patterns(y_test, y_pred, merged_test, output_file=None):
    """
    Analyze error patterns in misclassifications.
    
    Args:
        y_test: True labels
        y_pred: Predictions
        merged_test: Test data with original features
        output_file: Output file path
    
    Returns:
        dict: Error analysis results
    """
    if output_file is None:
        output_file = RESULTS_DIR / 'error_pattern_analysis.txt'
    
    misclassified = y_test != y_pred
    false_positives = (y_test == 0) & (y_pred == 1)
    false_negatives = (y_test == 1) & (y_pred == 0)
    
    analysis = {
        'total_misclassifications': misclassified.sum(),
        'false_positives': false_positives.sum(),
        'false_negatives': false_negatives.sum(),
        'patterns': {}
    }
    
    # Analyze by sentiment
    if 'textblob_sentiment' in merged_test.columns:
        sentiment_errors = merged_test[misclassified]['textblob_sentiment'].value_counts()
        analysis['patterns']['by_sentiment'] = sentiment_errors.to_dict()
        
        print("\n" + "="*80)
        print("ERROR PATTERNS BY SENTIMENT")
        print("="*80)
        for sentiment, count in sentiment_errors.items():
            pct = (count / misclassified.sum()) * 100
            print(f"  {sentiment.upper():10s}: {count:4d} errors ({pct:5.2f}% of all errors)")
    
    # Analyze by target
    target_errors = pd.Series(y_test[misclassified]).value_counts()
    analysis['patterns']['by_target'] = target_errors.to_dict()
    
    print("\n" + "="*80)
    print("ERROR PATTERNS BY TARGET")
    print("="*80)
    for target, count in target_errors.items():
        label = "Up (1)" if target == 1 else "Down (0)"
        pct = (count / misclassified.sum()) * 100
        print(f"  {label:10s}: {count:4d} errors ({pct:5.2f}% of all errors)")
    
    # Analyze false positives by sentiment
    if false_positives.sum() > 0 and 'textblob_sentiment' in merged_test.columns:
        fp_sentiment = merged_test[false_positives]['textblob_sentiment'].value_counts()
        print("\n" + "="*80)
        print("FALSE POSITIVES BY SENTIMENT")
        print("="*80)
        for sentiment, count in fp_sentiment.items():
            print(f"  {sentiment.upper():10s}: {count:4d} false positives")
        
        # Check for finance jargon in false positives
        if 'tweet' in merged_test.columns or 'Tweet' in merged_test.columns:
            tweet_col = 'tweet' if 'tweet' in merged_test.columns else 'Tweet'
            fp_tweets = merged_test[false_positives][tweet_col]
            
            # Check for $ symbols (finance jargon)
            dollar_count = fp_tweets.astype(str).str.contains(r'\$', regex=True).sum()
            print(f"\n  Tweets with $ symbol: {dollar_count} ({dollar_count/len(fp_tweets)*100:.1f}%)")
            
            if dollar_count > len(fp_tweets) * 0.3:
                analysis['patterns']['finance_jargon_issue'] = True
                print("  ⚠️  PATTERN DETECTED: Many false positives contain finance symbols ($)")
                print("     Model may be confusing neutral finance tweets as positive")
    
    # Analyze false negatives by sentiment
    if false_negatives.sum() > 0 and 'textblob_sentiment' in merged_test.columns:
        fn_sentiment = merged_test[false_negatives]['textblob_sentiment'].value_counts()
        print("\n" + "="*80)
        print("FALSE NEGATIVES BY SENTIMENT")
        print("="*80)
        for sentiment, count in fn_sentiment.items():
            print(f"  {sentiment.upper():10s}: {count:4d} false negatives")
    
    # Analyze by stock
    if 'stock name' in merged_test.columns:
        stock_errors = merged_test[misclassified]['stock name'].value_counts()
        print("\n" + "="*80)
        print("ERROR PATTERNS BY STOCK")
        print("="*80)
        for stock, count in stock_errors.head(10).items():
            pct = (count / misclassified.sum()) * 100
            print(f"  {stock:10s}: {count:4d} errors ({pct:5.2f}% of all errors)")
        analysis['patterns']['by_stock'] = stock_errors.to_dict()
    
    # Save analysis
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ERROR PATTERN ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Misclassifications: {misclassified.sum()}\n")
        f.write(f"False Positives: {false_positives.sum()}\n")
        f.write(f"False Negatives: {false_negatives.sum()}\n\n")
        
        f.write("Patterns:\n")
        for pattern_type, pattern_data in analysis['patterns'].items():
            f.write(f"\n{pattern_type}:\n")
            if isinstance(pattern_data, dict):
                for key, value in pattern_data.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {pattern_data}\n")
    
    print(f"\n✅ Saved error pattern analysis to: {output_file}")
    
    return analysis


def document_error_findings(analysis, output_file=None):
    """
    Document error findings in a structured format.
    
    Args:
        analysis: Error analysis results
        output_file: Output file path
    """
    if output_file is None:
        output_file = RESULTS_DIR / 'error_findings.md'
    
    findings = []
    
    # Finance jargon issue
    if analysis['patterns'].get('finance_jargon_issue', False):
        findings.append({
            'issue': 'Finance Jargon Confusion',
            'description': 'Model confuses neutral finance tweets as positive due to $ symbols',
            'impact': 'High false positive rate for neutral tweets with finance symbols',
            'recommendation': 'Consider feature engineering to handle finance symbols separately, or use domain-specific sentiment analysis'
        })
    
    # Sentiment-based patterns
    if 'by_sentiment' in analysis['patterns']:
        sentiment_errors = analysis['patterns']['by_sentiment']
        if 'neutral' in sentiment_errors and sentiment_errors['neutral'] > analysis['total_misclassifications'] * 0.4:
            findings.append({
                'issue': 'Neutral Sentiment Misclassification',
                'description': f"High error rate for neutral tweets ({sentiment_errors['neutral']} errors)",
                'impact': 'Model struggles with neutral sentiment classification',
                'recommendation': 'Consider adjusting sentiment thresholds or using ensemble methods'
            })
    
    # Write findings
    with open(output_file, 'w') as f:
        f.write("# Error Analysis Findings\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total Misclassifications: {analysis['total_misclassifications']}\n")
        f.write(f"- False Positives: {analysis['false_positives']}\n")
        f.write(f"- False Negatives: {analysis['false_negatives']}\n\n")
        
        f.write("## Identified Issues\n\n")
        for i, finding in enumerate(findings, 1):
            f.write(f"### Issue {i}: {finding['issue']}\n\n")
            f.write(f"**Description:** {finding['description']}\n\n")
            f.write(f"**Impact:** {finding['impact']}\n\n")
            f.write(f"**Recommendation:** {finding['recommendation']}\n\n")
    
    print(f"✅ Saved error findings to: {output_file}")
    
    return findings

