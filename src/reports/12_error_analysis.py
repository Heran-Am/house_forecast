import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# -------- paths --------
DATA_PATH = Path("data/processed/features_delta_v1.parquet")
OUT_DIR = Path("reports/error_analysis")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TIME = "date"
GROUP = "zipcode"
CITY = "city_full"
TARGET_Y = "y"
LAG1 = "lag_1"
DELTA_TARGET = "delta_y"

# must match your delta feature file columns
FEATURE_COLS = [
    "month", "year",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6",
    "inventory_lag1", "new_listings_lag1", "pending_sales_lag1", "homes_sold_lag1",
    "median_dom_lag1", "avg_sale_to_list_lag1",
    "median_list_price_lag1", "median_ppsf_lag1",
    "delta_1", "delta_3", "pct_change_1", "pct_change_3",
]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_delta_model(train_df, valid_df):
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[DELTA_TARGET].to_numpy(dtype=np.float32)

    X_valid = valid_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_valid = valid_df[DELTA_TARGET].to_numpy(dtype=np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
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
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=200,
        verbose_eval=False
    )
    return booster

def predict_price(booster, df_test):
    dtest = xgb.DMatrix(df_test[FEATURE_COLS].to_numpy(dtype=np.float32))

    # XGBoost 3.x uses iteration_range
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        y_pred_delta = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    else:
        y_pred_delta = booster.predict(dtest)

    # reconstruct price
    pred_price = df_test[LAG1].to_numpy(dtype=np.float32) + y_pred_delta
    return pred_price

def main():
    df = pd.read_parquet(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # We'll analyze on the same split style you used:
    # Train <= 2020, Valid = 2021, Test = 2022+
    train = df[df[TIME] <= "2020-12-31"].copy()
    valid = df[(df[TIME] >= "2021-01-31") & (df[TIME] <= "2021-12-31")].copy()
    test  = df[df[TIME] >= "2022-01-31"].copy()

    booster = train_delta_model(train, valid)

    y_true = test[TARGET_Y].to_numpy(dtype=np.float32)
    y_pred = predict_price(booster, test)
    y_naive = test[LAG1].to_numpy(dtype=np.float32)

    # overall
    overall = {
        "naive_mae": mean_absolute_error(y_true, y_naive),
        "model_mae": mean_absolute_error(y_true, y_pred),
        "naive_rmse": rmse(y_true, y_naive),
        "model_rmse": rmse(y_true, y_pred),
    }
    overall["mae_improvement"] = overall["naive_mae"] - overall["model_mae"]
    overall["rmse_improvement"] = overall["naive_rmse"] - overall["model_rmse"]

    pd.DataFrame([overall]).to_csv(OUT_DIR / "overall_metrics.csv", index=False)

    # add errors back to frame
    out = test[[TIME, GROUP, CITY, TARGET_Y, LAG1]].copy()
    out["y_pred"] = y_pred
    out["y_naive"] = y_naive
    out["abs_err_model"] = np.abs(out[TARGET_Y] - out["y_pred"])
    out["abs_err_naive"] = np.abs(out[TARGET_Y] - out["y_naive"])
    out["residual_model"] = out[TARGET_Y] - out["y_pred"]
    out["residual_naive"] = out[TARGET_Y] - out["y_naive"]

    out.to_parquet(OUT_DIR / "predictions_test_2022plus.parquet", index=False)

    # per-city leaderboard
    city_table = (
        out.groupby(CITY)
        .agg(
            rows=(TARGET_Y, "size"),
            model_mae=("abs_err_model", "mean"),
            naive_mae=("abs_err_naive", "mean"),
        )
        .reset_index()
    )
    city_table["mae_improvement"] = city_table["naive_mae"] - city_table["model_mae"]
    city_table = city_table.sort_values("mae_improvement", ascending=False)
    city_table.to_csv(OUT_DIR / "city_mae_leaderboard.csv", index=False)

    # plot: top/bottom cities
    top5 = city_table.head(5)
    bottom5 = city_table.tail(5)

    def bar_plot(df_bar, title, filename):
        plt.figure()
        plt.bar(df_bar[CITY], df_bar["mae_improvement"])
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel("MAE improvement vs naive ($)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=150)
        plt.close()

    bar_plot(top5, "Top 5 cities (best improvement)", "top5_city_improvement.png")
    bar_plot(bottom5, "Bottom 5 cities (worst improvement)", "bottom5_city_improvement.png")

    # plot: example time series for one city (hard) and one city (easy)
    # pick hard = lowest improvement, easy = highest improvement
    easy_city = city_table.iloc[0][CITY]
    hard_city = city_table.iloc[-1][CITY]

    def plot_city_series(city_name, filename):
        dfc = out[out[CITY] == city_name].sort_values(TIME)
        # pick one zipcode with many points
        zip_counts = dfc[GROUP].value_counts()
        z = int(zip_counts.index[0])
        s = dfc[dfc[GROUP] == z]

        plt.figure()
        plt.plot(s[TIME], s[TARGET_Y], label="Actual")
        plt.plot(s[TIME], s["y_naive"], label="Naive")
        plt.plot(s[TIME], s["y_pred"], label="Model")
        plt.title(f"{city_name} — ZIP {z} (Actual vs Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Median sale price ($)")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=150)
        plt.close()

    plot_city_series(easy_city, "example_easy_city_series.png")
    plot_city_series(hard_city, "example_hard_city_series.png")

    print("Saved error analysis to:", OUT_DIR)
    print("Overall improvement (MAE, RMSE):", overall["mae_improvement"], overall["rmse_improvement"])
    print("Easy city example:", easy_city)
    print("Hard city example:", hard_city)

if __name__ == "__main__":
    main()
