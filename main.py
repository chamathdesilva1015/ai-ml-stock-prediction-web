from data_utils import download_stock_data, engineer_features
from model_utils import train_rf
import json
import os
import joblib

with open("config.json") as f:
    config = json.load(f)

for ticker in config["tickers"]:
    df = download_stock_data(ticker, config["start_date"], config["end_date"])
    df = engineer_features(df)

    for horizon in config["forecast_horizons"]:
        y = df[f"Target_{horizon}"]
        rf_model, mae_rf, r2_rf = train_rf(df[config["features"]], y, config["rf_estimators"])
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_model, f"models/{ticker}_rf_{horizon}d.pkl")

        print(f"{ticker} {horizon}-day RF MAE: {mae_rf:.4f}, R2: {r2_rf:.4f}")
