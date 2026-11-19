
from pathlib import Path
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "monthly_category_sales.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "category_sales_forecast.csv"
)

# 2. Load monthly sales data
df = pd.read_csv(DATA_PATH)

df["Month"] = pd.to_datetime(df["Month"])

categories = df["Category"].unique()

all_forecasts = []

# 3. Train forecasting model for each category
for category in categories:

    print(f"\nTraining forecast model for: {category}")

    category_data = df[
        df["Category"] == category
    ].copy()

    category_data = category_data.set_index("Month")

    sales = category_data["Total Sales"].asfreq("MS")

    # 4. Build Holt-Winters model
    model = ExponentialSmoothing(
        sales,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    )

    fitted_model = model.fit()

    # 5. Forecast next 12 months
    forecast = fitted_model.forecast(12)

    # 6. Store forecast results
    forecast_df = pd.DataFrame({
        "Month": forecast.index,
        "Category": category,
        "Predicted Sales": forecast.values
    })

    all_forecasts.append(forecast_df)

    print("\nNext 12 months forecast:")
    print(forecast.round(2))

    print(
        "\nTotal annual forecast:",
        round(forecast.sum(), 2)
    )

# 7. Combine all category forecasts
final_forecast = pd.concat(
    all_forecasts,
    ignore_index=True
)

# 8. Export forecast results
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_forecast.to_csv(
    OUTPUT_PATH,
    index=False
)

# 9. Display annual forecast summary
summary = (
    final_forecast.groupby("Category")["Predicted Sales"]
    .sum()
    .reset_index()
)

summary = summary.rename(
    columns={"Predicted Sales": "Annual Forecast Sales"}
)

print("\n========== ANNUAL SALES FORECAST ==========")
print(summary.round(2))

print("\nTotal forecasted sales across all categories:")
print(round(final_forecast["Predicted Sales"].sum(), 2))

print("\nForecast exported successfully:")
print(OUTPUT_PATH)

print("\nTotal forecast records:", len(final_forecast))
