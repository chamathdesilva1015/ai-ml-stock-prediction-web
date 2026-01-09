# AI/ML-Powered Stock Prediction Application

An interactive AI/ML-powered stock forecasting application that applies supervised machine learning models to predict short-term stock price movements using historical market data and technical indicators.
This project was designed as an end-to-end system covering data ingestion, feature engineering, model training, evaluation, ensemble inference, and web-based visualization

# Project Overview

The application enables users to:

* Select one or more publicly traded stocks
* Choose a prediction horizon (1-day or 5-day)
* Generate model-based price forecasts
* Visualize historical stock price trends interactively

Predictions are generated using an ensemble of machine learning regression models trained on engineered technical features derived from historical price data
This project emphasizes machine learning fundamentals, software modularity, and practical deployment considerations, rather than financial trading performance.

# Machine Learning Methodology

# Models

* Random Forest Regressor
* Gradient Boosting Regressor

An ensemble strategy is used to combine predictions from multiple models by averaging their outputs, improving robustness and reducing variance.

# Feature Engineering

The model is trained on technical indicators including:

* Simple Moving Averages (SMA-10, SMA-50)
* Exponential Moving Average (EMA-20)
* Relative Strength Index (RSI-14)
* Daily returns
* Rolling volatility
* Lagged closing prices (1–3 days)

Target variables are defined for both 1-day and 5-day future closing prices.

# Training and Evaluation

* Time-aware train/test split (no shuffling)
* Hyperparameter tuning using GridSearchCV
* Model performance evaluated using Mean Absolute Error (MAE) and R² score
* Trained models are persisted to disk for reuse during inference

# System Architecture

The project is organized into modular components:

data_utils.py      - Data ingestion and feature engineering
model_utils.py     - Model training, tuning, and evaluation
train_models.py    - Offline model training and persistence
predict_utils.py   - Model loading and ensemble inference
app.py             - Streamlit-based user interface

This structure separates training from inference, improves maintainability, and supports future extensions such as additional models or data sources.

# Technology Stack

# Languages

* Python

# Libraries and Frameworks

* Streamlit
* scikit-learn
* pandas, NumPy
* yfinance
* TextBlob

# Machine Learning Techniques

* Supervised regression
* Ensemble learning
* Hyperparameter optimization
* Feature engineering


# How to Run the Project

# 1. Install dependencies

Create and activate a virtual environment, then install dependencies:
pip install -r requirements.txt

# 2. Train the models

This step downloads historical stock data, engineers features, trains models for each stock and forecast horizon, and saves them to the models/ directory
python train_models.py

# 3. Launch the application

The application will be available locally in your browser.
streamlit run app.py

# Limitations and Future Improvements

# Current Limitations

* Relies solely on historical price-based features
* Short prediction horizons (1-day and 5-day)
* Not intended for real-time or production trading use
* Local execution only (no cloud deployment)

# Potential Extensions

* REST API using FastAPI for model serving
* Frontend migration to React
* Model explainability using SHAP
* News sentiment integration for additional signals
* Cloud deployment using Docker and AWS

# Disclaimer

This project is for educational and exploratory purposes only.
It does not provide financial advice and should not be used for real-world trading decisions.

# Author
Chamath De Silva
Honours Computer Science (Level 3)
McMaster University
Email: chamathdesilva975@gmail.com