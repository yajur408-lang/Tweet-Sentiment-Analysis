"""
Deep learning models for sentiment classification with custom cost-aware loss.

Includes:
- Feed-forward (dense) neural network on averaged embeddings
- Shallow 1D CNN for text
- Simple vanilla RNN
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Deep learning models will be skipped.")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from .cost_sensitive_evaluation import (
    COST_MATRIX_TF, custom_sentiment_loss, compute_cost_score,
    compute_expected_cost, CLASS_NAMES
)
from .config import RANDOM_STATE

if TF_AVAILABLE:
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def prepare_embeddings_for_nn(embeddings: np.ndarray, method='mean'):
    """
    Prepare embeddings for neural network input.
    
    Args:
        embeddings: Word embeddings (n_samples, seq_len, embedding_dim) or averaged
        method: 'mean', 'max', or 'concat_mean_max'
    
    Returns:
        np.ndarray: Processed embeddings
    """
    if len(embeddings.shape) == 2:
        # Already averaged/processed
        return embeddings
    
    if method == 'mean':
        return np.mean(embeddings, axis=1)
    elif method == 'max':
        return np.max(embeddings, axis=1)
    elif method == 'concat_mean_max':
        mean_emb = np.mean(embeddings, axis=1)
        max_emb = np.max(embeddings, axis=1)
        return np.concatenate([mean_emb, max_emb], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def build_feedforward_nn(input_dim: int, n_classes: int = 3, 
                        hidden_dims: List[int] = [128, 64],
                        dropout_rate: float = 0.5,
                        use_custom_loss: bool = True):
    """
    Build feed-forward neural network on averaged embeddings.
    
    Args:
        input_dim: Input dimension (embedding size)
        n_classes: Number of classes (3 for sentiment)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        use_custom_loss: If True, use custom cost-aware loss
    
    Returns:
        keras.Model: Compiled model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Dense(hidden_dims[0], activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dropout(dropout_rate))
    
    # Hidden layers
    for dim in hidden_dims[1:]:
        model.add(layers.Dense(dim, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(n_classes, activation='softmax'))
    
    # Compile model
    if use_custom_loss:
        loss_fn = custom_sentiment_loss
        loss_name = 'custom_sentiment_loss'
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = 'categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"✅ Built Feed-forward NN with {loss_name}")
    return model


def build_cnn_1d(input_length: int, embedding_dim: int, vocab_size: int = None,
                n_classes: int = 3, filters: List[int] = [128, 64],
                kernel_sizes: List[int] = [3, 4, 5],
                dropout_rate: float = 0.5, use_pretrained_embeddings: bool = False,
                embedding_matrix: np.ndarray = None, use_custom_loss: bool = True):
    """
    Build shallow 1D CNN for text.
    
    Args:
        input_length: Maximum sequence length
        embedding_dim: Embedding dimension
        vocab_size: Vocabulary size (if using learned embeddings)
        n_classes: Number of classes
        filters: Number of filters per convolution
        kernel_sizes: List of kernel sizes for parallel convolutions
        dropout_rate: Dropout rate
        use_pretrained_embeddings: If True, use pretrained embedding matrix
        embedding_matrix: Pretrained embedding matrix (vocab_size, embedding_dim)
        use_custom_loss: If True, use custom cost-aware loss
    
    Returns:
        keras.Model: Compiled model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    # Input layer
    input_layer = layers.Input(shape=(input_length,))
    
    # Embedding layer
    if use_pretrained_embeddings and embedding_matrix is not None:
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False
        )(input_layer)
    else:
        # If we already have embeddings, just reshape
        if vocab_size is None:
            # Assume input is already embedded
            embedding = input_layer
            if len(input_layer.shape) == 2:
                embedding = layers.Reshape((input_length, embedding_dim))(input_layer)
        else:
            embedding = layers.Embedding(
                vocab_size, embedding_dim,
                input_length=input_length
            )(input_layer)
    
    # Parallel convolutions with different kernel sizes
    conv_outputs = []
    for kernel_size in kernel_sizes:
        conv = layers.Conv1D(filters[0], kernel_size, activation='relu')(embedding)
        pool = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(pool)
    
    # Concatenate parallel convolutions
    if len(conv_outputs) > 1:
        concat = layers.Concatenate()(conv_outputs)
    else:
        concat = conv_outputs[0]
    
    # Dense layers
    dense = layers.Dense(filters[1], activation='relu')(concat)
    dense = layers.Dropout(dropout_rate)(dense)
    
    # Output layer
    output = layers.Dense(n_classes, activation='softmax')(dense)
    
    model = models.Model(inputs=input_layer, outputs=output)
    
    # Compile model
    if use_custom_loss:
        loss_fn = custom_sentiment_loss
        loss_name = 'custom_sentiment_loss'
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = 'categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"✅ Built 1D CNN with {loss_name}")
    return model


def build_simple_rnn(input_length: int, embedding_dim: int, vocab_size: int = None,
                    n_classes: int = 3, hidden_units: int = 64,
                    dropout_rate: float = 0.5, use_pretrained_embeddings: bool = False,
                    embedding_matrix: np.ndarray = None, use_custom_loss: bool = True):
    """
    Build simple vanilla RNN (for learning purposes).
    
    Args:
        input_length: Maximum sequence length
        embedding_dim: Embedding dimension
        vocab_size: Vocabulary size (if using learned embeddings)
        n_classes: Number of classes
        hidden_units: Number of RNN hidden units
        dropout_rate: Dropout rate
        use_pretrained_embeddings: If True, use pretrained embedding matrix
        embedding_matrix: Pretrained embedding matrix
        use_custom_loss: If True, use custom cost-aware loss
    
    Returns:
        keras.Model: Compiled model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    # Input layer
    input_layer = layers.Input(shape=(input_length,))
    
    # Embedding layer
    if use_pretrained_embeddings and embedding_matrix is not None:
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False
        )(input_layer)
    else:
        if vocab_size is None:
            # Assume input is already embedded
            embedding = input_layer
            if len(input_layer.shape) == 2:
                embedding = layers.Reshape((input_length, embedding_dim))(input_layer)
        else:
            embedding = layers.Embedding(
                vocab_size, embedding_dim,
                input_length=input_length
            )(input_layer)
    
    # Simple RNN layer
    rnn = layers.SimpleRNN(hidden_units, return_sequences=False)(embedding)
    rnn = layers.Dropout(dropout_rate)(rnn)
    
    # Output layer
    output = layers.Dense(n_classes, activation='softmax')(rnn)
    
    model = models.Model(inputs=input_layer, outputs=output)
    
    # Compile model
    if use_custom_loss:
        loss_fn = custom_sentiment_loss
        loss_name = 'custom_sentiment_loss'
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = 'categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"✅ Built Simple RNN with {loss_name}")
    print("   Note: LSTM/GRU are typically better - this is for learning purposes")
    return model


def train_deep_model(model: 'keras.Model', X_train, y_train, X_val, y_val,
                    epochs: int = 20, batch_size: int = 32,
                    verbose: int = 1, early_stopping: bool = True):
    """
    Train a deep learning model.
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels (one-hot encoded)
        X_val: Validation features
        y_val: Validation labels (one-hot encoded)
        epochs: Number of epochs
        batch_size: Batch size
        verbose: Verbosity level
        early_stopping: If True, use early stopping
    
    Returns:
        keras.Model: Trained model
        keras.History: Training history
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    callbacks_list = []
    
    if early_stopping:
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        callbacks_list.append(early_stop)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=verbose
    )
    
    return model, history


def evaluate_deep_model(model: 'keras.Model', X_test, y_test, 
                       return_proba: bool = True) -> Dict[str, Any]:
    """
    Evaluate deep learning model with cost-aware metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        return_proba: If True, return probabilities
    
    Returns:
        dict: Evaluation results
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    # Predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Convert one-hot to integer labels
    y_test_int = np.argmax(y_test, axis=1)
    
    # Compute metrics
    cost = compute_cost_score(y_test_int, y_pred)
    expected_cost = compute_expected_cost(y_test_int, y_proba)
    
    results = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_proba if return_proba else None,
        'accuracy': accuracy_score(y_test_int, y_pred),
        'cost': cost,
        'expected_cost': expected_cost,
        'f1_macro': f1_score(y_test_int, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test_int, y_pred)
    }
    
    return results


def train_feedforward_nn_on_embeddings(X_train, y_train, X_test, y_test,
                                       embedding_method='mean',
                                       hidden_dims=[128, 64],
                                       dropout_rate=0.5,
                                       epochs=20, batch_size=32,
                                       use_custom_loss=True,
                                       validation_split=0.2):
    """
    Train feed-forward NN on averaged embeddings.
    
    Args:
        X_train: Training embeddings (n_samples, seq_len, embedding_dim) or averaged
        y_train: Training labels (strings or integers)
        X_test: Test embeddings
        y_test: Test labels
        embedding_method: Method to process embeddings ('mean', 'max', 'concat_mean_max')
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        use_custom_loss: If True, use custom cost-aware loss
        validation_split: Fraction of training data to use for validation
    
    Returns:
        dict: Model results
    """
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available. Skipping deep learning model.")
        return None
    
    print("\n📊 Training Feed-forward Neural Network...")
    
    # Prepare embeddings
    X_train_processed = prepare_embeddings_for_nn(X_train, method=embedding_method)
    X_test_processed = prepare_embeddings_for_nn(X_test, method=embedding_method)
    
    input_dim = X_train_processed.shape[1]
    
    # Encode labels
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    # Convert to one-hot
    y_train_onehot = to_categorical(y_train_encoded, num_classes=3)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=3)
    
    # Split training data for validation
    n_val = int(len(X_train_processed) * validation_split)
    X_val = X_train_processed[-n_val:]
    y_val = y_train_onehot[-n_val:]
    X_train_final = X_train_processed[:-n_val]
    y_train_final = y_train_onehot[:-n_val]
    
    # Build model
    model = build_feedforward_nn(
        input_dim=input_dim,
        n_classes=3,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_custom_loss=use_custom_loss
    )
    
    # Train model
    model, history = train_deep_model(
        model, X_train_final, y_train_final, X_val, y_val,
        epochs=epochs, batch_size=batch_size, verbose=1
    )
    
    # Evaluate model
    results = evaluate_deep_model(model, X_test_processed, y_test_onehot)
    results['history'] = history
    results['label_encoder'] = le
    
    return results


def create_model_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table for all models (classical ML + deep learning).
    
    Args:
        results: Dictionary of model results
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result.get('accuracy', np.nan),
            'Cost': result.get('cost', np.nan),
            'Expected Cost': result.get('expected_cost', np.nan),
            'F1-Macro': result.get('f1_macro', np.nan)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Cost')  # Sort by cost (lower is better)
    
    return df

