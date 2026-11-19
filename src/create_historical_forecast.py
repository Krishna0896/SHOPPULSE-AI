
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

HISTORICAL_PATH = (
    PROJECT_DIR / "data" / "processed" / "monthly_category_sales.csv"
)

FORECAST_PATH = (
    PROJECT_DIR / "reports" / "category_sales_forecast.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "reports" / "historical_and_forecast_sales.csv"
)

# Load historical sales
historical = pd.read_csv(HISTORICAL_PATH)
historical["Month"] = pd.to_datetime(historical["Month"])

historical = historical.rename(
    columns={"Total Sales": "Actual Sales"}
)

historical["Forecast Sales"] = None

historical = historical[
    ["Month", "Category", "Actual Sales", "Forecast Sales"]
]

# Load future forecast
forecast = pd.read_csv(FORECAST_PATH)
forecast["Month"] = pd.to_datetime(forecast["Month"])

forecast = forecast.rename(
    columns={"Predicted Sales": "Forecast Sales"}
)

forecast["Actual Sales"] = None

forecast = forecast[
    ["Month", "Category", "Actual Sales", "Forecast Sales"]
]

# Combine historical and forecast data
combined = pd.concat(
    [historical, forecast],
    ignore_index=True
)

combined = combined.sort_values(
    ["Category", "Month"]
).reset_index(drop=True)

# Export combined dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

combined.to_csv(OUTPUT_PATH, index=False)

print("========== HISTORICAL + FORECAST DATA ==========")

print("\nDataset shape:", combined.shape)

print("\nDate range:")
print(combined["Month"].min(), "to", combined["Month"].max())

print("\nRecords by category:")
print(combined.groupby("Category").size())

print("\nHistorical and forecast sample:")
print(combined.head(5).to_string(index=False))

print("\nLast forecast records:")
print(combined.tail(5).to_string(index=False))

print("\nTotal records:", len(combined))

print("\nFile exported successfully:")
print(OUTPUT_PATH)
