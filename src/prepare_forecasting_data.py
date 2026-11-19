
from pathlib import Path
import pandas as pd

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "superstore_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "monthly_category_sales.csv"
)

# 2. Load existing cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# 3. Convert Order Date into datetime format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# 4. Aggregate monthly sales by category
monthly_sales = (
    df.groupby(
        [
            pd.Grouper(key="Order Date", freq="MS"),
            "Category"
        ]
    )["Sales"]
    .sum()
    .reset_index()
)

# 5. Rename columns
monthly_sales.columns = [
    "Month",
    "Category",
    "Total Sales"
]

# 6. Sort the records
monthly_sales = monthly_sales.sort_values(
    ["Month", "Category"]
)

# 7. Export monthly sales dataset
monthly_sales.to_csv(OUTPUT_PATH, index=False)

# 8. Display results
print("\nMonthly category sales:")
print(monthly_sales.head(15))

print("\nDataset shape:", monthly_sales.shape)

print("\nCategories:")
print(monthly_sales["Category"].unique())

print("\nDate range:")
print(monthly_sales["Month"].min())
print(monthly_sales["Month"].max())

print("\nMonthly sales dataset exported:")
print(OUTPUT_PATH)
