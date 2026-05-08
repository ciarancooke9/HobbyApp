# src/features/build_features.py

import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------
    # Numeric columns
    # --------------------
    num_cols = [
        "Trip_Distance_km", "Passenger_Count",
        "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate",
        "Trip_Duration_Minutes"
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Fill numeric missing
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # --------------------
    # Categorical
    # --------------------
    cat_cols = [
        "Time_of_Day", "Day_of_Week",
        "Traffic_Conditions", "Weather"
    ]

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # --------------------
    # Feature Engineering
    # --------------------
    df["Distance_Cost"] = df["Trip_Distance_km"] * df["Per_Km_Rate"]
    df["Time_Cost"] = df["Trip_Duration_Minutes"] * df["Per_Minute_Rate"]

    df["Avg_Speed"] = df["Trip_Distance_km"] / (df["Trip_Duration_Minutes"] / 60 + 1e-5)

    df["Is_Peak"] = df["Time_of_Day"].isin(["Morning", "Evening"]).astype(int)
    df["Is_Weekend"] = (df["Day_of_Week"] == "Weekend").astype(int)
    df["High_Traffic"] = (df["Traffic_Conditions"] == "High").astype(int)
    df["Bad_Weather"] = df["Weather"].isin(["Rain", "Snow"]).astype(int)

    return df
