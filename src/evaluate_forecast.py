
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "monthly_category_sales.csv"
)

ACCURACY_PATH = (
    PROJECT_DIR / "reports" / "forecast_evaluation.csv"
)

COMPARISON_PATH = (
    PROJECT_DIR / "reports" / "forecast_vs_actual_2017.csv"
)

df = pd.read_csv(DATA_PATH)
df["Month"] = pd.to_datetime(df["Month"])

accuracy_results = []
comparison_results = []

for category in df["Category"].unique():

    print(f"\nEvaluating category: {category}")

    category_data = (
        df[df["Category"] == category]
        .sort_values("Month")
        .set_index("Month")
    )

    sales = category_data["Total Sales"].asfreq("MS")

    # Training: first 36 months
    train = sales.iloc[:36]

    # Testing: final 12 months
    test = sales.iloc[36:]

    # Train Holt-Winters model
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    )

    fitted_model = model.fit()

    # Predict the 12 test months
    predictions = fitted_model.forecast(12)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test, predictions)
    rmse = mean_squared_error(test, predictions) ** 0.5

    accuracy_results.append({
        "Category": category,
        "MAE": mae,
        "RMSE": rmse
    })

    # Monthly actual vs predicted comparison
    comparison = pd.DataFrame({
        "Month": test.index,
        "Category": category,
        "Actual Sales": test.values,
        "Forecast Sales": predictions.values
    })

    comparison["Variance"] = (
        comparison["Actual Sales"] -
        comparison["Forecast Sales"]
    )

    comparison["Absolute Error"] = (
        comparison["Variance"].abs()
    )

    comparison_results.append(comparison)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

# Export accuracy metrics
accuracy_df = pd.DataFrame(accuracy_results)

ACCURACY_PATH.parent.mkdir(parents=True, exist_ok=True)

accuracy_df.to_csv(ACCURACY_PATH, index=False)

# Export monthly comparison
comparison_df = pd.concat(
    comparison_results,
    ignore_index=True
)

comparison_df.to_csv(COMPARISON_PATH, index=False)

print("\n========== FORECAST EVALUATION ==========")
print(accuracy_df.round(2).to_string(index=False))

print("\n========== ACTUAL VS FORECAST SAMPLE ==========")
print(comparison_df.head(10).round(2).to_string(index=False))

print("\nFiles exported successfully:")
print(ACCURACY_PATH)
print(COMPARISON_PATH)

print("\nTotal comparison records:", len(comparison_df))
