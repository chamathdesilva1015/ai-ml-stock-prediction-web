import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)

def train_model(X, y, model_type="RF", n_estimators=200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if model_type == "RF":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    elif model_type == "GB":
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError("Unknown model type")
    
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="r2")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{model_type} model - MAE: {mae:.4f}, R2: {r2:.4f}")
    return best_model

def save_model(model, path):
    joblib.dump(model, path)
