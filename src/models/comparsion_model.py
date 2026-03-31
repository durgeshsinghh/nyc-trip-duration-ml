# Install once (if not installed)
# pip install lazypredict

import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("data/processed/train.csv")

# ----------------------------
# SAMPLE (VERY IMPORTANT ⚡)
# ----------------------------
df = df.sample(n=50000, random_state=42)  # use 50K rows

# ----------------------------
# Split Features & Target
# ----------------------------
target = "trip_duration"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# LazyPredict
# ----------------------------
reg = LazyRegressor(
    verbose=0,
    ignore_warnings=True,
    predictions=False
)

models, _ = reg.fit(X_train, X_test, y_train, y_test)

# ----------------------------
# Show Results
# ----------------------------
print(models)

# Save results (optional 🔥)
models.to_csv("reports/model_comparison.csv")