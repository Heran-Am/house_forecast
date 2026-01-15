import pandas as pd

PATH = "data/raw/HouseTS.csv"

def main():
    df = pd.read_csv(PATH)
    print("Shape (rows, cols):", df.shape)
    print("\nColumns:\n", df.columns.tolist())

    print("\nFirst 5 rows:\n", df.head(5))

    print("\nDtypes:\n", df.dtypes)

    missing = df.isna().sum().sort_values(ascending=False)
    print("\nTop 15 missing columns:\n", missing.head(15))

if __name__ == "__main__":
    main()
