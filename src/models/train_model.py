import pandas as pd
import joblib
import pathlib
import sys
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error



# TRAIN + COMPARE MODELS

def train_and_compare(X_train, X_test, y_train, y_test):

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Model_Comparison")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            n_jobs=2,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=2,
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))

            print(f"{name} RMSE: {rmse:.4f}")

            # MLflow logging
            mlflow.log_param("model", name)
            mlflow.log_metric("RMSE", rmse)

            mlflow.sklearn.log_model(model, "model")

            results[name] = rmse
            trained_models[name] = model


    # SELECT BEST MODEL

    best_model_name = min(results, key=results.get)
    best_model = trained_models[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print(f"Best RMSE: {results[best_model_name]:.4f}")

    return best_model, best_model_name, results



# MAIN

def main():

    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parent.parent.parent

    input_path = pathlib.Path(sys.argv[1])

    train_data = pd.read_csv(input_path / "train.csv")

    print("Data loaded")

    # SAMPLE DATA FOR SPEED
    train_data = train_data.sample(n=100000, random_state=42)

    target = "trip_duration"

    # Remove non-numeric columns
    train_data = train_data.select_dtypes(include=["int64", "float64"])

    X = train_data.drop(target, axis=1)
    y = train_data[target]

    # TRAIN-TEST SPLIT (IMPORTANT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training started...")

    best_model, best_model_name, results = train_and_compare(
        X_train, X_test, y_train, y_test
    )
    # SAVE BEST MODEL
    model_path = home_dir / "models"
    model_path.mkdir(exist_ok=True)

    joblib.dump(best_model, model_path / "model.joblib")

    print(f"\nBest model ({best_model_name}) saved at: {model_path / 'model.joblib'}")


if __name__ == "__main__":
    main()