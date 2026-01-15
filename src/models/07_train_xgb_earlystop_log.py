import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

FEATURES_PATH = Path("data/processed/features_h1.parquet")

TIME = "date"
GROUP = "zipcode"
TARGET = "y"

FEATURE_COLS = [
    "month", "year",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6"
]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # Load features
    df = pd.read_parquet(FEATURES_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # Train / Valid / Test split by time
    train = df[df[TIME] <= "2020-12-31"].copy()
    valid = df[(df[TIME] >= "2021-01-31") & (df[TIME] <= "2021-12-31")].copy()
    test  = df[df[TIME] >= "2022-01-31"].copy()

    print("Rows - train/valid/test:", len(train), len(valid), len(test))
    assert train[TIME].max() < valid[TIME].min(), "Train overlaps valid!"
    assert valid[TIME].max() < test[TIME].min(),  "Valid overlaps test!"

    # Prepare matrices
    X_train = train[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_valid = valid[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_test  = test[FEATURE_COLS].to_numpy(dtype=np.float32)

    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_valid = valid[TARGET].to_numpy(dtype=np.float32)
    y_test  = test[TARGET].to_numpy(dtype=np.float32)

    # Naive baseline (lag_1) on TEST
    y_pred_naive = test["lag_1"].to_numpy(dtype=np.float32)
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = rmse(y_test, y_pred_naive)

    print("\n=== Naive (lag_1) on TEST ===")
    print(f"MAE : {mae_naive:,.2f}")
    print(f"RMSE: {rmse_naive:,.2f}")

    # Log target
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # DMatrix containers
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dvalid = xgb.DMatrix(X_valid, label=y_valid_log)
    dtest  = xgb.DMatrix(X_test)

    # Params (regularized, stable)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "min_child_weight": 10,
        "seed": 42,
    }

    # Train with early stopping
    evals = [(dtrain, "train"), (dvalid, "valid")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=50
    )

    best_iter = getattr(booster, "best_iteration", None)

    print("\n=== XGBoost (core train) + early stopping + log(target) ===")
    print("Num boosted rounds:", booster.num_boosted_rounds())
    print("Best iteration:", best_iter)

    # Predict using best iteration trees
    if best_iter is not None:
    # XGBoost 3.x uses iteration_range, ntree_limit is removed
      y_pred_log = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    else:
      y_pred_log = booster.predict(dtest)


    # Convert back to dollars
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    r = rmse(y_test, y_pred)

    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {r:,.2f}")

    print("\n=== Improvement vs Naive (TEST) ===")
    print(f"MAE improvement : {mae_naive - mae:,.2f} (positive = better)")
    print(f"RMSE improvement: {rmse_naive - r:,.2f} (positive = better)")

if __name__ == "__main__":
    main()
