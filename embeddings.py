"""
Word embeddings module using Word2Vec and TF-IDF
Implements data leakage prevention by training only on training data
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
tqdm.pandas()
from .config import W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT


def clean_text(text):
    """Clean and tokenize text for Word2Vec."""
    text = re.sub(r'[^a-zA-Z ]', '', str(text)).lower()
    return word_tokenize(text)


def prepare_tokens(merged):
    """
    Clean and tokenize tweets for Word2Vec.
    
    Args:
        merged: Dataframe with tweet column (can be 'Tweet', 'tweet', etc.)
    
    Returns:
        Dataframe with 'tokens' column
    """
    print("Preparing tokens for Word2Vec...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    # Handle different column name variations
    tweet_col = None
    for col_name in ['tweet', 'Tweet', 'TWEET', 'text', 'Text', 'TEXT']:
        if col_name in merged.columns:
            tweet_col = col_name
            break
    
    if tweet_col is None:
        raise KeyError(f"Tweet column not found. Available columns: {merged.columns.tolist()}")
    
    merged['tokens'] = merged[tweet_col].progress_apply(clean_text)
    print("[OK] Tokens prepared")
    return merged


def train_word2vec_on_train(train_tokens):
    """
    Train Word2Vec model ONLY on training data to prevent data leakage.
    
    Args:
        train_tokens: List of tokenized tweets from training set only
    
    Returns:
        Trained Word2Vec model
    """
    print("Training Word2Vec model on TRAINING DATA ONLY (preventing data leakage)...")
    sentences = [t for t in train_tokens if t]
    
    w2v_model = Word2Vec(
        sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        sg=1
    )
    
    print(f"[OK] Word2Vec trained on training data! Vocabulary size: {len(w2v_model.wv)}")
    return w2v_model


def create_w2v_embeddings(tokens_series, w2v_model):
    """
    Create average word vectors for tweets using a pre-trained Word2Vec model.
    This function can be used on both train and test data.
    
    Args:
        tokens_series: Series of tokenized tweets
        w2v_model: Pre-trained Word2Vec model (trained on training data only)
    
    Returns:
        DataFrame with Word2Vec embeddings
    """
    def tweet_vector(tokens, model):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
    
    embeddings = tokens_series.progress_apply(lambda x: tweet_vector(x, w2v_model))
    embeddings_df = pd.DataFrame(
        embeddings.tolist(),
        columns=[f'w2v_{i}' for i in range(w2v_model.vector_size)]
    )
    
    return embeddings_df


def train_tfidf_on_train(train_tweets, max_features=5000):
    """
    Train TF-IDF vectorizer ONLY on training data to prevent data leakage.
    
    Args:
        train_tweets: Series of tweet text from training set only
        max_features: Maximum number of features to extract
    
    Returns:
        Fitted TfidfVectorizer
    """
    print(f"Training TF-IDF vectorizer on TRAINING DATA ONLY (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(train_tweets.astype(str))
    print(f"[OK] TF-IDF trained on training data! Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    return vectorizer


def create_tfidf_embeddings(tweets_series, tfidf_vectorizer):
    """
    Transform tweets to TF-IDF features using a pre-trained vectorizer.
    This function can be used on both train and test data.
    
    Args:
        tweets_series: Series of tweet text
        tfidf_vectorizer: Pre-trained TfidfVectorizer (fitted on training data only)
    
    Returns:
        DataFrame with TF-IDF embeddings
    """
    tfidf_matrix = tfidf_vectorizer.transform(tweets_series.astype(str))
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )
    return tfidf_df

