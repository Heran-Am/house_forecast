import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_PATH = Path("data/processed/features_h1.parquet")

TIME = "date"
GROUP = "zipcode"
TARGET = "y"

TRAIN_END = "2021-12-31"

FEATURE_COLS = [
    "month", "year",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6"
]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = pd.read_parquet(FEATURES_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # --- Time split (no leakage) ---
    train = df[df[TIME] <= TRAIN_END].copy()
    test  = df[df[TIME] >  TRAIN_END].copy()

    # Sanity check for leakage (time overlap)
    train_max = train[TIME].max()
    test_min  = test[TIME].min()
    print("Train max date:", train_max)
    print("Test min date :", test_min)
    assert train_max < test_min, "Leakage risk: train overlaps test in time!"

    # --- Naive baseline from features (should match your 03_baseline_naive) ---
    # Because lag_1 is literally last month’s y.
    y_true_test = test[TARGET].values
    y_pred_naive = test["lag_1"].values

    mae_naive = mean_absolute_error(y_true_test, y_pred_naive)
    rmse_naive = rmse(y_true_test, y_pred_naive)

    print("\n=== Baseline from features (lag_1) ===")
    print(f"Test MAE : {mae_naive:,.2f}")
    print(f"Test RMSE: {rmse_naive:,.2f}")

    # --- Ridge model ---
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values

    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET].values

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r = rmse(y_test, y_pred)

    print("\n=== Ridge Regression (with scaling) ===")
    print("Features:", FEATURE_COLS)
    print("Train rows:", len(train), "| Test rows:", len(test))
    print(f"Test MAE : {mae:,.2f}")
    print(f"Test RMSE: {r:,.2f}")

    # Improvement vs naive
    print("\n=== Improvement vs Naive ===")
    print(f"MAE improvement : {mae_naive - mae:,.2f} (positive = better)")
    print(f"RMSE improvement: {rmse_naive - r:,.2f} (positive = better)")

    # Optional: city breakdown (nice for report)
    if "city_full" in test.columns:
        tmp = test.copy()
        tmp["pred_ridge"] = y_pred
        tmp["abs_err"] = (tmp[TARGET] - tmp["pred_ridge"]).abs()

        mae_by_city = tmp.groupby("city_full")["abs_err"].mean().sort_values(ascending=False)

        print("\nTop 5 cities by Ridge MAE (hardest):")
        print(mae_by_city.head(5).to_string())

        print("\nBottom 5 cities by Ridge MAE (easiest):")
        print(mae_by_city.tail(5).to_string())

if __name__ == "__main__":
    main()
