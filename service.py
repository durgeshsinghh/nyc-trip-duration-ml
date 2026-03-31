from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("models/model.joblib")


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class InputData(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    pickup_hour: int
    pickup_day: int
    pickup_weekday: int
    pickup_month: int


# -----------------------------
# SAME FEATURE ENGINEERING AS TRAINING
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def create_features(df):

    # Distance features
    df["distance_haversine"] = haversine(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    df["distance_manhattan"] = (
        abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
        abs(df["pickup_longitude"] - df["dropoff_longitude"])
    )

    # Direction
    df["direction"] = np.arctan2(
        df["dropoff_latitude"] - df["pickup_latitude"],
        df["dropoff_longitude"] - df["pickup_longitude"]
    )

    # Log distance
    df["distance_log"] = np.log1p(df["distance_haversine"])

    # Coordinate diff
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # ⚠️ PCA placeholders (IMPORTANT)
    df["pickup_pca0"] = 0
    df["pickup_pca1"] = 0
    df["dropoff_pca0"] = 0
    df["dropoff_pca1"] = 0

    return df


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    df = create_features(df)

    # Keep same column order as training
    expected_cols = [
        'vendor_id', 'passenger_count',
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'store_and_fwd_flag',
        'pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month',
        'distance_haversine', 'distance_manhattan', 'direction',
        'distance_log', 'lat_diff', 'lon_diff',
        'pickup_pca0', 'pickup_pca1',
        'dropoff_pca0', 'dropoff_pca1'
    ]

    df = df[expected_cols]

    prediction = model.predict(df)[0]

    return {"trip_duration": float(prediction)}