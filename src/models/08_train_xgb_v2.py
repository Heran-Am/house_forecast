import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

FEATURES_PATH = Path("data/processed/features_h1_v2.parquet")

TIME = "date"
GROUP = "zipcode"
TARGET = "y"

FEATURE_COLS = [
    # calendar
    "month", "year",
    # price history
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6",
    # lagged market signals (leak-safe)
    "inventory_lag1",
    "new_listings_lag1",
    "pending_sales_lag1",
    "homes_sold_lag1",
    "median_dom_lag1",
    "avg_sale_to_list_lag1",
    "median_list_price_lag1",
    "median_ppsf_lag1",
]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = pd.read_parquet(FEATURES_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # Train / Valid / Test split
    train = df[df[TIME] <= "2020-12-31"].copy()
    valid = df[(df[TIME] >= "2021-01-31") & (df[TIME] <= "2021-12-31")].copy()
    test  = df[df[TIME] >= "2022-01-31"].copy()

    print("Rows - train/valid/test:", len(train), len(valid), len(test))
    assert train[TIME].max() < valid[TIME].min()
    assert valid[TIME].max() < test[TIME].min()

    X_train = train[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_valid = valid[FEATURE_COLS].to_numpy(dtype=np.float32)
    X_test  = test[FEATURE_COLS].to_numpy(dtype=np.float32)

    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_valid = valid[TARGET].to_numpy(dtype=np.float32)
    y_test  = test[TARGET].to_numpy(dtype=np.float32)

    # Naive baseline (lag_1)
    y_pred_naive = test["lag_1"].to_numpy(dtype=np.float32)
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = rmse(y_test, y_pred_naive)

    print("\n=== Naive (lag_1) on TEST ===")
    print(f"MAE : {mae_naive:,.2f}")
    print(f"RMSE: {rmse_naive:,.2f}")

    # Log target
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    dtrain = xgb.DMatrix(X_train, label=y_train_log, feature_names=FEATURE_COLS)
    dvalid = xgb.DMatrix(X_valid, label=y_valid_log, feature_names=FEATURE_COLS)
    dtest  = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "min_child_weight": 10,
        "seed": 42,
    }

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

    print("\n=== XGBoost v2 (market lags) + early stopping + log(target) ===")
    print("Num boosted rounds:", booster.num_boosted_rounds())
    print("Best iteration:", best_iter)

    # XGBoost 3.x: use iteration_range
    if best_iter is not None:
        y_pred_log = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    else:
        y_pred_log = booster.predict(dtest)

    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    r = rmse(y_test, y_pred)

    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {r:,.2f}")

    print("\n=== Improvement vs Naive (TEST) ===")
    print(f"MAE improvement : {mae_naive - mae:,.2f} (positive = better)")
    print(f"RMSE improvement: {rmse_naive - r:,.2f} (positive = better)")

    # Feature importance (gain)
    try:
        score = booster.get_score(importance_type="gain")
        top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 features by gain:")
        for k, v in top:
            print(f"{k:>22} : {v:.4f}")
    except Exception as e:
        print("Could not compute feature importance:", e)

if __name__ == "__main__":
    main()
