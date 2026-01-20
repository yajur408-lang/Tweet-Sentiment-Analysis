"""
Feature engineering module for technical indicators
"""
import pandas as pd
import numpy as np
from .utils import detect_close_column


def compute_moving_averages(merged, close_col, windows=None):
    """
    Compute moving averages for different windows.
    
    Args:
        merged: Dataframe with price data
        close_col: Name of closing price column
        windows: List of window sizes (default: [5, 10, 20])
    
    Returns:
        Dataframe with moving average features
    """
    from .config import MA_WINDOWS
    
    if windows is None:
        windows = MA_WINDOWS
    
    for window in windows:
        merged[f'ma_{window}'] = merged.groupby('stock name')[close_col].rolling(
            window=window
        ).mean().reset_index(0, drop=True)
    
    return merged


def compute_rsi(merged, close_col, window=14):
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        merged: Dataframe with price data
        close_col: Name of closing price column
        window: RSI window size (default: 14)
    
    Returns:
        Dataframe with RSI feature
    """
    delta = merged.groupby('stock name')[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    merged['rsi'] = 100 - (100 / (1 + rs))
    return merged


def compute_volatility(merged, window=30):
    """
    Compute rolling volatility (standard deviation of returns).
    
    Args:
        merged: Dataframe with price_change_pct column
        window: Volatility window size (default: 30)
    
    Returns:
        Dataframe with volatility feature
    """
    merged['volatility'] = merged.groupby('stock name')['price_change_pct'].rolling(
        window=window
    ).std().reset_index(0, drop=True)
    return merged


def compute_volume_features(merged, window=10):
    """
    Compute volume-related features.
    
    Args:
        merged: Dataframe with volume column
        window: Moving average window for volume (default: 10)
    
    Returns:
        Dataframe with volume features
    """
    if 'volume' in merged.columns:
        merged['volume_ma'] = merged.groupby('stock name')['volume'].rolling(
            window=window
        ).mean().reset_index(0, drop=True)
        merged['volume_ratio'] = merged['volume'] / merged['volume_ma']
    return merged


def compute_technical_indicators(merged, close_col=None):
    """
    Compute all technical indicators.
    
    IMPORTANT: All indicators use only PAST data (backward-looking).
    - Moving averages: rolling window looking backward
    - RSI: uses past price changes
    - Volatility: rolling std of past returns
    - Price momentum: diff() looks at previous row
    
    No future data or target variable is used in feature creation.
    
    Args:
        merged: Dataframe with price data
        close_col: Name of closing price column (auto-detected if None)
    
    Returns:
        Dataframe with all technical indicators
    """
    from .config import MA_WINDOWS, RSI_WINDOW, VOLATILITY_WINDOW, VOLUME_MA_WINDOW
    
    if close_col is None:
        close_col = detect_close_column(merged)
    
    print("Computing technical indicators (using only past data)...")
    
    # Sort by stock and date for time series calculations
    merged = merged.sort_values(['stock name', 'date']).reset_index(drop=True)
    
    # Moving averages - uses rolling window (backward-looking)
    merged = compute_moving_averages(merged, close_col, MA_WINDOWS)
    
    # Price momentum - diff() looks at previous row (backward-looking)
    merged['price_change'] = merged.groupby('stock name')[close_col].diff()
    merged['price_change_pct'] = merged.groupby('stock name')[close_col].pct_change()
    
    # RSI - uses past price changes (backward-looking)
    merged = compute_rsi(merged, close_col, RSI_WINDOW)
    
    # Volatility - rolling std of past returns (backward-looking)
    merged = compute_volatility(merged, VOLATILITY_WINDOW)
    
    # High-Low spread - uses current day's high/low (available at prediction time)
    if 'high' in merged.columns and 'low' in merged.columns:
        merged['hl_spread'] = (merged['high'] - merged['low']) / merged[close_col]
    
    # Volume features - rolling mean of past volume (backward-looking)
    merged = compute_volume_features(merged, VOLUME_MA_WINDOW)
    
    print("[OK] Technical indicators created!")
    print(f"  - Moving averages: {MA_WINDOWS} (backward-looking)")
    print(f"  - RSI (window={RSI_WINDOW}, backward-looking)")
    print(f"  - Volatility (window={VOLATILITY_WINDOW}, backward-looking)")
    print(f"  - Volume features (backward-looking)")
    print("  [SECURE] All features use only data available at prediction time")
    
    return merged

