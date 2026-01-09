import streamlit as st
import pandas as pd
import yfinance as yf
from data_utils import engineer_features, get_features_and_targets, download_stock_data
from predict_utils import load_model, ensemble_predict
import os

st.set_page_config(page_title="AI/ML-Powered Stock Prediction Application", layout="wide")
st.title("AI/ML-Powered Stock Prediction Application")

tickers = st.multiselect(
    "Select Stock Tickers",
    ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"],
    default=["AAPL"]
)

forecast_horizon = st.selectbox("Forecast Horizon (days ahead)", [1, 5])

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

for ticker in tickers:
    st.subheader(f"{ticker} Prediction")

    # Download historical stock data
    df = download_stock_data(ticker)
    
    # Generate technical features for the model
    try:
        df = engineer_features(df)
    except Exception as e:
        st.error(f"Error processing features for {ticker}: {e}")
        continue

    # Extract features and targets for prediction
    try:
        X, y1, y5 = get_features_and_targets(df)
    except Exception as e:
        st.error(f"Error preparing features/targets for {ticker}: {e}")
        continue

    last_rows = X.iloc[-forecast_horizon:]

    rf_path = f"models/{ticker}_rf_{forecast_horizon}.pkl"
    gb_path = f"models/{ticker}_gb_{forecast_horizon}.pkl"
    
    if not os.path.exists(rf_path) or not os.path.exists(gb_path):
        st.warning(f"Models for {ticker} and horizon {forecast_horizon} not found. Please run train_models.py first.")
        continue

    # Load trained ML models
    try:
        rf_model = load_model(rf_path)
        gb_model = load_model(gb_path)
    except Exception as e:
        st.error(f"Error loading models for {ticker}: {e}")
        continue

    # Make ensemble predictions
    try:
        pred = ensemble_predict([rf_model, gb_model], last_rows)
    except Exception as e:
        st.error(f"Error making predictions for {ticker}: {e}")
        continue

    for i, p in enumerate(pred, 1):
        st.write(f"Day +{i} Prediction: **${p:.2f}**")

    # Display historical stock prices
    days = st.slider(f"{ticker} - Days to plot", min_value=30, max_value=len(df), value=min(100, len(df)))
    st.line_chart(df["Close"].tail(days))
