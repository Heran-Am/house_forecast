import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

DATA_PATH = Path("data/processed/features_delta_v2_seasonality.parquet")

TIME = "date"
GROUP = "zipcode"
TRUE_Y = "y"
BASELINE = "lag_1"
TARGET_DELTA = "delta_y"

FEATURE_COLS = [
    "month", "year", "month_sin", "month_cos",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6",
    "inventory_lag1",
    "new_listings_lag1",
    "pending_sales_lag1",
    "homes_sold_lag1",
    "median_dom_lag1",
    "avg_sale_to_list_lag1",
    "median_list_price_lag1",
    "median_ppsf_lag1",
    "delta_1", "delta_3",
    "pct_change_1", "pct_change_3",
]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_delta_model(train_df, valid_df):
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_DELTA].to_numpy(dtype=np.float32)

    X_valid = valid_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_valid = valid_df[TARGET_DELTA].to_numpy(dtype=np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=FEATURE_COLS)

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

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return booster, getattr(booster, "best_iteration", None)

def predict_delta(booster, best_iter, X_test):
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)
    if best_iter is not None:
        return booster.predict(dtest, iteration_range=(0, best_iter + 1))
    return booster.predict(dtest)

def main():
    df = pd.read_parquet(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)
    df["year_int"] = df[TIME].dt.year

    years = sorted(df["year_int"].unique())
    test_years = [y for y in years if y >= 2018]

    results = []
    for test_year in test_years:
        valid_year = test_year - 1

        train_df = df[df["year_int"] <= (valid_year - 1)].copy()
        valid_df = df[df["year_int"] == valid_year].copy()
        test_df  = df[df["year_int"] == test_year].copy()

        if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
            continue

        y_true = test_df[TRUE_Y].to_numpy(dtype=np.float32)
        y_naive = test_df[BASELINE].to_numpy(dtype=np.float32)

        mae_naive = mean_absolute_error(y_true, y_naive)
        rmse_naive = rmse(y_true, y_naive)

        booster, best_iter = train_delta_model(train_df, valid_df)

        X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        delta_pred = predict_delta(booster, best_iter, X_test)
        y_pred = y_naive + delta_pred

        mae = mean_absolute_error(y_true, y_pred)
        r = rmse(y_true, y_pred)

        results.append({
            "test_year": test_year,
            "best_iter": best_iter,
            "naive_mae": mae_naive,
            "model_mae": mae,
            "mae_improvement": mae_naive - mae,
            "naive_rmse": rmse_naive,
            "model_rmse": r,
            "rmse_improvement": rmse_naive - r,
        })

        print(f"\n=== TEST YEAR {test_year} ===")
        print(f"Best iter: {best_iter}")
        print(f"Naive MAE {mae_naive:,.2f} | RMSE {rmse_naive:,.2f}")
        print(f"Model MAE {mae:,.2f} | RMSE {r:,.2f}")
        print(f"Improvement: MAE {mae_naive - mae:,.2f} | RMSE {rmse_naive - r:,.2f}")

    res = pd.DataFrame(results).sort_values("test_year")
    print("\n=== SUMMARY (Seasonality walk-forward) ===")
    print(res.to_string(index=False))
    print("\nAverage improvement:")
    print(f"MAE  avg improvement : {res['mae_improvement'].mean():,.2f}")
    print(f"RMSE avg improvement: {res['rmse_improvement'].mean():,.2f}")

if __name__ == "__main__":
    main()
