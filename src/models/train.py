import pandas as pd
import joblib
from pathlib import Path
import json

from src.features.build_features import build_features
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("data/raw/taxi_trip_pricing.csv")

# Feature engineering
df = build_features(df)

X = df.drop(columns=["Trip_Price"])
y = df["Trip_Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model

with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/xgboost_model.joblib")