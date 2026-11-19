
from pathlib import Path
import pandas as pd

# 1. Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Define the cleaned dataset path
DATA_PATH = PROJECT_DIR / "data" / "processed" / "superstore_cleaned.csv"

# 3. Load the dataset
df = pd.read_csv(DATA_PATH)

# 4. Inspect the dataset
print("Dataset loaded successfully!")
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# 5. Select input features
features = [
    "Quantity",
    "Discount",
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Ship Mode",
    "Order Year",
    "Order Month",
    "Order Quarter"
]

# 6. Create X (independent variables)
X = df[features]

# 7. Create y (dependent variable)
y = df["Sales"]

# 8. Inspect X and y
print("\nFeature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

print("\nInput features:")
print(X.head())

print("\nTarget values:")
print(y.head())


# 9. Import train_test_split
from sklearn.model_selection import train_test_split

# 10. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 11. Display dataset sizes
print("\nTraining and Testing Dataset Sizes")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 12. Identify numerical and categorical columns
numerical_features = [
    "Quantity",
    "Discount",
    "Order Year",
    "Order Month",
    "Order Quarter"
]

categorical_features = [
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Ship Mode"
]

# 13. Create categorical encoder
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# 14. Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing configuration created successfully!")
print(preprocessor)


# 15. Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)

# 16. Transform testing data
X_test_processed = preprocessor.transform(X_test)

# 17. Inspect transformed data
print("\nPreprocessing completed!")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)


# 18. Import Linear Regression
from sklearn.linear_model import LinearRegression

# 19. Create the complete ML pipeline
linear_regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

print("\nLinear Regression pipeline created successfully!")
print(linear_regression_pipeline)


# 20. Train the Linear Regression model
print("\nTraining Linear Regression model...")

linear_regression_pipeline.fit(X_train, y_train)

print("Linear Regression model trained successfully!")



# 21. Generate predictions on test data
y_pred = linear_regression_pipeline.predict(X_test)

print("\nFirst 10 Sales Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Sales Values:")
print(y_test.iloc[:10].values)


# 22. Import evaluation metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 23. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

# 24. Display model performance
print("\nLinear Regression Model Evaluation")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# 25. Evaluate training performance
y_train_pred = linear_regression_pipeline.predict(X_train)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 26. Display training vs testing performance
print("\nTraining vs Testing Performance")

print("Training MAE:", train_mae)
print("Testing MAE:", mae)

print("Training R2:", train_r2)
print("Testing R2:", r2)


# 27. Import DummyRegressor
from sklearn.dummy import DummyRegressor

# 28. Create baseline model
baseline_model = DummyRegressor(strategy="mean")

# 29. Train baseline model
baseline_model.fit(X_train, y_train)

# 30. Generate baseline predictions
baseline_predictions = baseline_model.predict(X_test)

# 31. Evaluate baseline model
baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

# 32. Compare baseline with Linear Regression
print("\nBaseline Model Evaluation")

print("Baseline MAE:", baseline_mae)
print("Baseline R2:", baseline_r2)

print("\nModel Comparison")

print("Linear Regression MAE:", mae)
print("Baseline MAE:", baseline_mae)

print("Linear Regression R2:", r2)
print("Baseline R2:", baseline_r2)


# 33. Create prediction analysis DataFrame

prediction_analysis = X_test.copy()

prediction_analysis.insert(
    0,
    "Prediction ID",
    X_test.index
)

# Add actual and predicted Sales
prediction_analysis["Actual Sales"] = y_test.values
prediction_analysis["Predicted Sales"] = y_pred

# Calculate prediction errors
prediction_analysis["Prediction Error"] = (
    prediction_analysis["Predicted Sales"]
    - prediction_analysis["Actual Sales"]
)

prediction_analysis["Absolute Error"] = (
    prediction_analysis["Prediction Error"].abs()
)

# Calculate percentage error safely
prediction_analysis["Absolute Percentage Error"] = (
    prediction_analysis["Absolute Error"]
    / prediction_analysis["Actual Sales"].abs().replace(0, float("nan"))
) * 100

# 34. Export prediction analysis
OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "linear_regression_predictions.csv"
)

prediction_analysis.to_csv(OUTPUT_PATH, index=False)

print("\nPrediction analysis exported successfully!")
print("Output location:", OUTPUT_PATH)

print("\nFirst 10 prediction comparisons:")
print(
    prediction_analysis[
        [
            "Actual Sales",
            "Predicted Sales",
            "Prediction Error",
            "Absolute Error"
        ]
    ].head(10).round(2)
)

print("\nTotal prediction records:", len(prediction_analysis))











