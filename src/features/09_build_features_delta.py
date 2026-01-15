import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("data/processed/features_h1_v2.parquet")
OUT_PATH = Path("data/processed/features_delta_v1.parquet")

TIME = "date"
GROUP = "zipcode"

# We already have these columns in v2:
BASE_FEATURES = [
    "month", "year",
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
]

def main():
    df = pd.read_parquet(IN_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # Delta target: y(t) - y(t-1)
    df["delta_y"] = df["y"] - df["lag_1"]

    # Momentum features (also leak-safe because they use only past lags)
    df["delta_1"] = df["lag_1"] - df.groupby(GROUP)["lag_1"].shift(1)  # y(t-1)-y(t-2)
    df["delta_3"] = df["lag_1"] - df["lag_3"]

    # Avoid divide-by-zero with small epsilon
    eps = 1e-6
    df["pct_change_1"] = df["delta_1"] / (df.groupby(GROUP)["lag_1"].shift(1) + eps)
    df["pct_change_3"] = df["delta_3"] / (df["lag_3"] + eps)

    feature_cols = BASE_FEATURES + ["delta_1", "delta_3", "pct_change_1", "pct_change_3"]

    df_out = df.dropna(subset=feature_cols + ["delta_y"]).copy()
    keep_cols = [TIME, GROUP, "city_full", "y",  "delta_y"] + feature_cols
    df_out = df_out[keep_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Shape:", df_out.shape)
    print("Features:", len(feature_cols))
    print("Date range:", df_out[TIME].min(), "→", df_out[TIME].max())

if __name__ == "__main__":
    main()
