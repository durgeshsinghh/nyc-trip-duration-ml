import pandas as pd
import numpy as np
from features_definations import (
    haversine_array,
    dummy_manhattan_distance,
    bearing_array
)
def test_feature_build(df):

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month

    df["distance_haversine"] = haversine_array(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    df["distance_manhattan"] = dummy_manhattan_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    df["direction"] = bearing_array(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    df["distance_log"] = np.log1p(df["distance_haversine"])

    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

    if "trip_duration" in df.columns:
        df = df[(df["trip_duration"] > 60) & (df["trip_duration"] < 3600)]

    if df["store_and_fwd_flag"].dtype == "object":
        df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return df
import pathlib
import pandas as pd
from features_definations import build_features

if __name__ == "__main__":

    # ----------------------------
    # 1. SET PATHS
    # ----------------------------
    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir / "data" / "raw" / "train.csv"
    test_path = home_dir / "data" / "raw" / "test.csv"

    # ----------------------------
    # 2. LOAD DATA
    # ----------------------------
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("Data loaded")

    # ----------------------------
    # 3. FEATURE ENGINEERING
    # ----------------------------
    train_data, test_data = build_features(train_data, test_data)

    print("Features created:", train_data.shape)

    # ----------------------------
    # 4. SAVE DATA
    # ----------------------------
    output_path = home_dir / "data" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(output_path / "train.csv", index=False)
    test_data.to_csv(output_path / "test.csv", index=False)

    print("Data saved successfully")