import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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

    train = df[df[TIME] <= TRAIN_END].copy()
    test  = df[df[TIME] >  TRAIN_END].copy()

    # Leakage check
    assert train[TIME].max() < test[TIME].min()

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    # Naive baseline (lag_1)
    y_pred_naive = test["lag_1"].values
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = rmse(y_test, y_pred_naive)

    print("=== Baseline (lag_1) ===")
    print(f"Test MAE : {mae_naive:,.2f}")
    print(f"Test RMSE: {rmse_naive:,.2f}")

    # XGBoost model (solid starter settings)
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r = rmse(y_test, y_pred)

    print("\n=== XGBoost ===")
    print("Train rows:", len(train), "| Test rows:", len(test))
    print(f"Test MAE : {mae:,.2f}")
    print(f"Test RMSE: {r:,.2f}")

    print("\n=== Improvement vs Naive ===")
    print(f"MAE improvement : {mae_naive - mae:,.2f} (positive = better)")
    print(f"RMSE improvement: {rmse_naive - r:,.2f} (positive = better)")

    # Feature importance (quick look)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\nTop feature importances:")
    for i in order[:8]:
        print(f"{FEATURE_COLS[i]:>12} : {importances[i]:.4f}")

if __name__ == "__main__":
    main()
