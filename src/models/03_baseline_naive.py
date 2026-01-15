import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

DATA_PATH = Path("data/processed/housets.parquet")

TARGET = "median_sale_price"
GROUP = "zipcode"
TIME = "date"

# We'll do a time-based split:
# Train: up to 2021-12-31
# Test:  2022-01-01 to 2023-12-31
TRAIN_END = "2021-12-31"

def main():
    df = pd.read_parquet(DATA_PATH)

    # Ensure proper types
    df[TIME] = pd.to_datetime(df[TIME])

    # Build naive prediction: previous month's value per zipcode
    df = df.sort_values([GROUP, TIME]).reset_index(drop=True)
    df["y_true"] = df[TARGET]
    df["y_pred_naive"] = df.groupby(GROUP)[TARGET].shift(1)

    # Drop first month per zipcode (no previous value)
    df_model = df.dropna(subset=["y_true", "y_pred_naive"]).copy()

    # Split by time (no leakage)
    train = df_model[df_model[TIME] <= TRAIN_END]
    test = df_model[df_model[TIME] > TRAIN_END]

    # Evaluate on test only (the honest score)
    y_true = test["y_true"]
    y_pred = test["y_pred_naive"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("=== Naive Baseline: predict next month = last month ===")
    print("Target:", TARGET)
    print("Train end:", TRAIN_END)
    print("Train rows:", len(train), "| Test rows:", len(test))
    print(f"Test MAE : {mae:,.2f}")
    print(f"Test RMSE: {rmse:,.2f}")

    # Optional: sanity check by city_full
    if "city_full" in test.columns:
        mae_by_city = (
            test.assign(abs_err=(test["y_true"] - test["y_pred_naive"]).abs())
                .groupby("city_full")["abs_err"]
                .mean()
                .sort_values(ascending=False)
        )
        print("\nTop 5 cities by MAE (hardest):")
        print(mae_by_city.head(5).to_string())

        print("\nBottom 5 cities by MAE (easiest):")
        print(mae_by_city.tail(5).to_string())

if __name__ == "__main__":
    main()
