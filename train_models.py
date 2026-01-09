# train_models.py

import os
import pandas as pd
from data_utils import engineer_features, get_features_and_targets, download_stock_data
from model_utils import train_model, save_model

tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
os.makedirs("models", exist_ok=True)

for ticker in tickers:
    print(f"\nProcessing {ticker}...")

    # Download stock data (or overwrite existing)
    df = download_stock_data(ticker)

    # Feature engineering
    try:
        df = engineer_features(df)
    except Exception as e:
        print(f"Error in feature engineering for {ticker}: {e}")
        continue

    # Extract features and targets
    try:
        X, y1, y5 = get_features_and_targets(df)
    except Exception as e:
        print(f"Error preparing features/targets for {ticker}: {e}")
        continue

    print("Training 1-day models...")
    rf_model_1 = train_model(X, y1, model_type="RF")
    gb_model_1 = train_model(X, y1, model_type="GB")
    save_model(rf_model_1, f"models/{ticker}_rf_1.pkl")
    save_model(gb_model_1, f"models/{ticker}_gb_1.pkl")

    print("Training 5-day models...")
    rf_model_5 = train_model(X, y5, model_type="RF")
    gb_model_5 = train_model(X, y5, model_type="GB")
    save_model(rf_model_5, f"models/{ticker}_rf_5.pkl")
    save_model(gb_model_5, f"models/{ticker}_gb_5.pkl")

print("\nAll models trained and saved successfully.")
