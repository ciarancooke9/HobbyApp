# src/models/predict.py

import pandas as pd
import joblib
from pathlib import Path

from src.features.build_features import build_features

# --------------------
# Paths
# --------------------
MODEL_PATH = Path("models/full_pipeline.joblib")
INPUT_PATH = Path("data/raw/new_data.csv")   # replace with your file
OUTPUT_PATH = Path("data/processed/predictions.csv")

# --------------------
# Load model
# --------------------
pipeline = joblib.load(MODEL_PATH)

# --------------------
# Load new data
# --------------------
df = pd.read_csv(INPUT_PATH)

# IMPORTANT: apply same features
df = build_features(df)

# --------------------
# Predict
# --------------------
predictions = pipeline.predict(df)

df["Predicted_Trip_Price"] = predictions

# --------------------
# Save output
# --------------------
OUTPUT_PATH.parent.mkdir(exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Predictions saved!")
