
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "superstore_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "reports" / "business_impact_analysis.csv"
)

df = pd.read_csv(DATA_PATH)

# Business dimensions to analyze
dimensions = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment"
]

results = []

for dimension in dimensions:

    analysis = (
        df.groupby(dimension)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Average_Discount=("Discount", "mean"),
            Order_Count=("Order ID", "nunique")
        )
        .reset_index()
        .rename(columns={dimension: "Dimension_Value"})
    )

    analysis["Dimension"] = dimension

    analysis["Profit_Margin"] = (
        analysis["Total_Profit"] / analysis["Total_Sales"]
    )

    results.append(analysis)

final_analysis = pd.concat(results, ignore_index=True)

final_analysis = final_analysis[
    [
        "Dimension",
        "Dimension_Value",
        "Total_Sales",
        "Total_Profit",
        "Total_Quantity",
        "Average_Discount",
        "Order_Count",
        "Profit_Margin"
    ]
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_analysis.to_csv(OUTPUT_PATH, index=False)

print("========== BUSINESS IMPACT ANALYSIS ==========")
print(final_analysis.round(3).to_string(index=False))

print("\nAnalysis exported successfully:")
print(OUTPUT_PATH)
