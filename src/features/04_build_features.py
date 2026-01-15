import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/housets.parquet")
OUT_PATH = Path("data/processed/features_h1.parquet")  # h1 = 1-month ahead

TARGET = "median_sale_price"
GROUP = "zipcode"
TIME = "date"

def main():
    df = pd.read_parquet(DATA_PATH)
    df[TIME] = pd.to_datetime(df[TIME])

    # Sort for time series
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)

    # Target
    df["y"] = df[TARGET]

    # Calendar features (safe)
    df["month"] = df[TIME].dt.month
    df["year"] = df[TIME].dt.year  # already exists but keep consistent

    # Lag features (safe: uses past values only)
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby(GROUP)["y"].shift(lag)

    # Rolling features (safe: shift first, then rolling)
    df["roll_mean_3"] = (
        df.groupby(GROUP)["y"]
          .shift(1)
          .rolling(3)
          .mean()
    )
    df["roll_mean_6"] = (
        df.groupby(GROUP)["y"]
          .shift(1)
          .rolling(6)
          .mean()
    )

    # Drop rows that don’t have enough history
    feature_cols = ["month", "year", "lag_1", "lag_3", "lag_6", "lag_12", "roll_mean_3", "roll_mean_6"]
    df_feat = df.dropna(subset=feature_cols + ["y"]).copy()

    # Keep only what we need
    keep_cols = [TIME, GROUP, "city_full", "y"] + feature_cols
    df_feat = df_feat[keep_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(OUT_PATH, index=False)

    print("Saved features to:", OUT_PATH)
    print("Feature table shape:", df_feat.shape)
    print("Date range:", df_feat[TIME].min(), "→", df_feat[TIME].max())

if __name__ == "__main__":
    main()
