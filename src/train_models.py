import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X = pd.get_dummies(X, drop_first=True)
    print(f"Data loaded: {df.shape}")
    return X, y

def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'model': model,
            'RMSE':  round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'MAE':   round(mean_absolute_error(y_test, y_pred), 2),
            'R2':    round(r2_score(y_test, y_pred), 4)
        }
        print(f"{name} — R2: {results[name]['R2']}")

    return results, X_test, y_test

def save_best_model(results):
    best = max(results, key=lambda x: results[x]['R2'])
    print(f"\nBest model: {best} (R2={results[best]['R2']})")

    with open('outputs/best_model_xgboost.pkl', 'wb') as f:
        pickle.dump(results[best]['model'], f)
    print("Model saved to outputs/")

if __name__ == "__main__":
    X, y = load_data('data/validated_clean_train.csv')
    results, X_test, y_test = train_all_models(X, y)
    save_best_model(results)