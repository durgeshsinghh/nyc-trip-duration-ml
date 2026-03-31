import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# =========================
# DISTANCE FUNCTIONS
# =========================

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371

    lat = lat2 - lat1
    lng = lng2 - lng1

    d = np.sin(lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2) ** 2
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    return (
        haversine_array(lat1, lng1, lat1, lng2) +
        haversine_array(lat1, lng1, lat2, lng1)
    )


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta)

    return np.degrees(np.arctan2(y, x))


# =========================
# DATETIME FEATURES
# =========================

def date_time_features_fix(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month

    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return df


# =========================
# DISTANCE FEATURES
# =========================

def create_dist_features(df):

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

    # LOG FEATURE
    df["distance_log"] = np.log1p(df["distance_haversine"])

    # COORD DIFFERENCE
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

    return df


# =========================
# PCA FEATURES (IMPORTANT)
# =========================

def create_pca_features(train_df, test_df):

    coords = np.vstack((
        train_df[["pickup_latitude", "pickup_longitude"]],
        train_df[["dropoff_latitude", "dropoff_longitude"]],
        test_df[["pickup_latitude", "pickup_longitude"]],
        test_df[["dropoff_latitude", "dropoff_longitude"]],
    ))

    pca = PCA().fit(coords)

    # TRAIN
    train_df["pickup_pca0"] = pca.transform(train_df[["pickup_latitude", "pickup_longitude"]])[:, 0]
    train_df["pickup_pca1"] = pca.transform(train_df[["pickup_latitude", "pickup_longitude"]])[:, 1]

    train_df["dropoff_pca0"] = pca.transform(train_df[["dropoff_latitude", "dropoff_longitude"]])[:, 0]
    train_df["dropoff_pca1"] = pca.transform(train_df[["dropoff_latitude", "dropoff_longitude"]])[:, 1]

    # TEST
    test_df["pickup_pca0"] = pca.transform(test_df[["pickup_latitude", "pickup_longitude"]])[:, 0]
    test_df["pickup_pca1"] = pca.transform(test_df[["pickup_latitude", "pickup_longitude"]])[:, 1]

    test_df["dropoff_pca0"] = pca.transform(test_df[["dropoff_latitude", "dropoff_longitude"]])[:, 0]
    test_df["dropoff_pca1"] = pca.transform(test_df[["dropoff_latitude", "dropoff_longitude"]])[:, 1]

    return train_df, test_df


# =========================
# CLEANING
# =========================

def clean_data(df):

    if "trip_duration" in df.columns:
        df = df[(df["trip_duration"] > 60) & (df["trip_duration"] < 3600)]

    return df


# =========================
# FINAL PIPELINE
# =========================

def build_features(train_df, test_df):

    # datetime
    train_df = date_time_features_fix(train_df)
    test_df = date_time_features_fix(test_df)

    # distance
    train_df = create_dist_features(train_df)
    test_df = create_dist_features(test_df)

    # PCA
    train_df, test_df = create_pca_features(train_df, test_df)

    # cleaning
    train_df = clean_data(train_df)

    # DROP NON-NUMERIC
    drop_cols = ["id", "pickup_datetime", "dropoff_datetime"]

    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    return train_df, test_df


# =========================
# TEST RUN
# =========================

if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv", nrows=1000)
    test = pd.read_csv("data/raw/test.csv", nrows=1000)

    train, test = build_features(train, test)

    print(train.head())
    print("Features created successfully")