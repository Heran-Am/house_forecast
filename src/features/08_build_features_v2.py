import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/housets.parquet")
OUT_PATH = Path("data/processed/features_h1_v2.parquet")

TIME = "date"
GROUP = "zipcode"
TARGET = "median_sale_price"

# Extra market signals to lag (must exist in housets.parquet)
EXTRA_COLS = [
    "inventory",
    "new_listings",
    "pending_sales",
    "homes_sold",
    "median_dom",
    "avg_sale_to_list",
    "median_list_price",
    "median_ppsf",
]

def main():
    df = pd.read_parquet(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # Target
    df["y"] = df[TARGET]

    # Calendar features
    df["month"] = df[TIME].dt.month
    df["year"] = df[TIME].dt.year

    # Price-history features
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby(GROUP)["y"].shift(lag)

    df["roll_mean_3"] = df.groupby(GROUP)["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df.groupby(GROUP)["y"].shift(1).rolling(6).mean()

    # Lagged market signals (leak-safe)
    for col in EXTRA_COLS:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby(GROUP)[col].shift(1)

    feature_cols = [
        "month", "year",
        "lag_1", "lag_3", "lag_6", "lag_12",
        "roll_mean_3", "roll_mean_6",
    ] + [f"{c}_lag1" for c in EXTRA_COLS if c in df.columns]

    # Drop rows without required history/signals
    df_feat = df.dropna(subset=feature_cols + ["y"]).copy()

    keep_cols = [TIME, GROUP, "city_full", "y"] + feature_cols
    df_feat = df_feat[keep_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Shape:", df_feat.shape)
    print("Feature columns:", len(feature_cols))
    print("Date range:", df_feat[TIME].min(), "→", df_feat[TIME].max())

if __name__ == "__main__":
    main()
