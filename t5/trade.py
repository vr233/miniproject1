import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "trade_sheet.csv"
OUTPUT_FILE = "trades_clean.csv"


def clean_trade_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())

    df = df.drop_duplicates().dropna()

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.strip().str.lower()
    else:
        df["side"] = np.where(df.get("profit_loss", 0) >= 0, "buy", "sell")

    numeric_cols = [
        "buy_price",
        "sell_price",
        "quantity",
        "holding_days",
        "brokerage",
        "profit_loss",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [col for col in ["buy_price", "sell_price", "quantity"] if col in df.columns]
    if not required_cols:
        raise ValueError("The CSV does not contain the required trade columns: buy_price, sell_price, quantity")

    df = df.dropna(subset=required_cols)
    df = df[(df["buy_price"] > 0) & (df["sell_price"] > 0) & (df["quantity"] > 0)]

    df["spread"] = df["sell_price"] - df["buy_price"]
    df["mid_price"] = (df["buy_price"] + df["sell_price"]) / 2
    df["imbalance"] = (df["spread"] / df["mid_price"]).fillna(0)

    return df


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["buy_price", "sell_price", "quantity", "spread"]
    features = df[feature_cols].copy()

    mean = features.mean()
    std = features.std(ddof=0).replace(0, 1)
    standardized = (features - mean) / std

    print("\nScaled data:\n", standardized.head())
    return standardized


def main() -> None:
    df = clean_trade_data(INPUT_FILE)

    df["mid_price"].plot(title="Trade Mid Price")
    plt.show()

    df["side"].value_counts().plot.bar(title="Buy vs Sell")
    plt.show()

    standardized = standardize_features(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nData cleaning completed!")
    print(f"Cleaned dataset saved successfully to {OUTPUT_FILE}.")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()