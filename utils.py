import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate various performance metrics for predictions
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
    
    Returns:
        Dictionary containing various metrics
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")
    
    if len(actual) == 0:
        return {}
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Direction accuracy (percentage of correct directional predictions)
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        direction_accuracy = 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r_squared': r_squared,
        'accuracy': direction_accuracy
    }

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists and has data
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Boolean indicating if symbol is valid
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        return not hist.empty
    except:
        return False

def get_stock_info(symbol: str) -> Optional[Dict]:
    """
    Get basic information about a stock
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with stock information or None if not found
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    except:
        return None

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    return df

def preprocess_data(data: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """
    Preprocess stock data for modeling
    
    Args:
        data: Raw stock data
        fill_missing: Whether to fill missing values
    
    Returns:
        Preprocessed DataFrame
    """
    df = data.copy()
    
    # Remove any duplicate dates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    # Fill missing values if requested
    if fill_missing:
        # Forward fill followed by backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    return df

def calculate_volatility(prices: np.ndarray, window: int = 20) -> float:
    """
    Calculate rolling volatility
    
    Args:
        prices: Array of prices
        window: Rolling window size
    
    Returns:
        Volatility value
    """
    if len(prices) < window:
        return np.std(prices)
    
    returns = np.diff(np.log(prices))
    return np.std(returns[-window:]) * np.sqrt(252)  # Annualized volatility

def detect_outliers(prices: np.ndarray, method: str = 'iqr') -> np.ndarray:
    """
    Detect outliers in price data
    
    Args:
        prices: Array of prices
        method: Method to use ('iqr' or 'zscore')
    
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (prices < lower_bound) | (prices > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((prices - np.mean(prices)) / np.std(prices))
        return z_scores > 3
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format a value as currency
    
    Args:
        value: Numeric value
        currency: Currency code
    
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def calculate_returns(prices: np.ndarray, method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from price data
    
    Args:
        prices: Array of prices
        method: 'simple' or 'log'
    
    Returns:
        Array of returns
    """
    if method == 'simple':
        return np.diff(prices) / prices[:-1]
    elif method == 'log':
        return np.diff(np.log(prices))
    else:
        raise ValueError("Method must be 'simple' or 'log'")

def get_market_hours() -> Dict[str, str]:
    """
    Get market hours for major exchanges
    
    Returns:
        Dictionary with market hours
    """
    return {
        'NYSE': '9:30 AM - 4:00 PM ET',
        'NASDAQ': '9:30 AM - 4:00 PM ET',
        'LSE': '8:00 AM - 4:30 PM GMT',
        'TSE': '9:00 AM - 3:00 PM JST',
        'HKEX': '9:30 AM - 4:00 PM HKT'
    }
