import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/features_delta_v1.parquet")
OUT_PATH = Path("data/processed/features_delta_v2_seasonality.parquet")

TIME = "date"
MONTH = "month"

def main():
    df = pd.read_parquet(IN_PATH)
    df[TIME] = pd.to_datetime(df[TIME])

    # month should already be 1..12, but ensure it is int
    df[MONTH] = df[MONTH].astype(int)

    # seasonality features: circular encoding
    df["month_sin"] = np.sin(2 * np.pi * df[MONTH] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df[MONTH] / 12.0)

    df.to_parquet(OUT_PATH, index=False)

    feature_cols = [c for c in df.columns if c not in ["date", "zipcode", "city_full", "y"]]
    print("Saved:", OUT_PATH)
    print("Shape:", df.shape)
    print("Feature columns:", len(feature_cols))
    print("Date range:", df[TIME].min(), "→", df[TIME].max())
    print("Added:", ["month_sin", "month_cos"])

if __name__ == "__main__":
    main()
