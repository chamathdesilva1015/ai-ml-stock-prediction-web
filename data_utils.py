import yfinance as yf
import pandas as pd
import os
from textblob import TextBlob

def download_stock_data(ticker, start="2020-01-01", end=None):
    """Download adjusted stock prices and return a DataFrame with 'Close' column."""
    os.makedirs("data", exist_ok=True)
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    df = df[['Close']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df.reset_index(inplace=True)
    df.to_csv(f"data/{ticker}_data.csv", index=False)

    return df


def compute_RSI(series, period=14):
    """Compute Relative Strength Index (RSI) for a series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df):
    """Add technical indicators and lagged features to the stock DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    
    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column not found in df.columns: {df.columns.tolist()}")

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI_14"] = compute_RSI(df["Close"], 14)
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    
    for lag in range(1, 4):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    
    df["Target_1"] = df["Close"].shift(-1)
    df["Target_5"] = df["Close"].shift(-5)
    df = df.dropna()

    return df


def get_features_and_targets(df):
    """Return numeric features X and targets y1, y5 for predictions."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_cols].drop(columns=['Target_1', 'Target_5'], errors='ignore')
    y1 = df['Target_1']
    y5 = df['Target_5']
    return X, y1, y5


def fetch_news_sentiment(text):
    """Return polarity of text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity
