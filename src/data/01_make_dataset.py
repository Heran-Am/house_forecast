import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/HouseTS.csv")
OUT_PATH = Path("data/processed/housets.parquet")

def main():
    print("Loading:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)

    # 1) Parse date properly
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2) ZIP codes: keep as string (preserves leading zeros)
    #    We format to 5 digits because US ZIPs are 5-digit in this dataset context.
    df["zipcode"] = df["zipcode"].astype(int).astype(str).str.zfill(5)

    # 3) Sort for time-series work
    df = df.sort_values(["zipcode", "date"]).reset_index(drop=True)

    # 4) Basic sanity checks
    print("\n--- Sanity checks ---")
    print("Rows, Cols:", df.shape)
    print("Date range:", df["date"].min(), "→", df["date"].max())
    print("Unique ZIP codes:", df["zipcode"].nunique())
    print("Unique city_full:", df["city_full"].nunique())

    # Check for any invalid dates
    bad_dates = df["date"].isna().sum()
    print("Bad (unparsed) dates:", bad_dates)

    # Missing values summary (only those > 0)
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print("\nColumns with missing values (>0):")
    if len(miss) == 0:
        print("None ✅")
    else:
        print(miss.head(30))

    # 5) Write to Parquet (fast + smaller than CSV)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print("\nSaved processed dataset to:", OUT_PATH)

if __name__ == "__main__":
    main()
