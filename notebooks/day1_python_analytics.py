from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. PROJECT PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

for folder in [PROCESSED_DIR, REPORTS_DIR, CHARTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================
print("\n========== LOADING DATA ==========")
df = pd.read_csv(RAW_FILE)
print("Dataset loaded successfully")

# Trim extra spaces from column names

df.columns = df.columns.str.strip()

# =========================
# 3. BASIC DATA INSPECTION
# =========================
print("\n========== DATA OVERVIEW ==========")
print("Shape (rows, columns):", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nDataset information:")
df.info()

# =========================
# 4. MISSING VALUES
# =========================
print("\n========== MISSING VALUES ==========")
missing_counts = df.isna().sum()
missing_percent = (df.isna().mean() * 100).round(2)
missing_report = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent
}).sort_values("missing_count", ascending=False)
print(missing_report)
missing_report.to_csv(REPORTS_DIR / "missing_values_report.csv")

# =========================
# 5. DUPLICATES
# =========================
print("\n========== DUPLICATES ==========")
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)
# Remove exact duplicate rows only
if duplicate_count:
    df = df.drop_duplicates().copy()
print("Shape after duplicate removal:", df.shape)

# =========================
# 6. DATE CONVERSION
# =========================
print("\n========== DATE PREPARATION ==========")
for date_col in ["Order Date", "Ship Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        print(f"Converted {date_col} to datetime")

# =========================
# 7. DESCRIPTIVE STATISTICS
# =========================
print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe(include="all").T)
df.describe(include="all").T.to_csv(REPORTS_DIR / "descriptive_statistics.csv")

# =========================
# 8. NUMERIC BUSINESS KPIs
# =========================
print("\n========== BUSINESS KPIs ==========")
expected_numeric = ["Sales", "Profit", "Quantity", "Discount"]
available_numeric = [col for col in expected_numeric if col in df.columns]
kpi_summary = df[available_numeric].agg(["sum", "mean", "min", "max"]).T if available_numeric else pd.DataFrame()
print(kpi_summary)
kpi_summary.to_csv(REPORTS_DIR / "numeric_kpi_summary.csv")

# =========================
# 9. CATEGORY / SUB-CATEGORY ANALYSIS
# =========================
print("\n========== CATEGORY ANALYSIS ==========")
if "Category" in df.columns and "Sales" in df.columns:
    category_sales = df.groupby("Category", dropna=False)["Sales"].sum().sort_values(ascending=False)
    print("Sales by category:\n", category_sales)
    category_sales.to_csv(REPORTS_DIR / "sales_by_category.csv")

if {"Sub-Category", "Sales"}.issubset(df.columns):
    subcategory_sales = df.groupby("Sub-Category", dropna=False)["Sales"].sum().sort_values(ascending=False)
    print("\nSales by sub-category:\n", subcategory_sales)
    subcategory_sales.to_csv(REPORTS_DIR / "sales_by_subcategory.csv")

if {"Category", "Profit"}.issubset(df.columns):
    category_profit = df.groupby("Category", dropna=False)["Profit"].sum().sort_values(ascending=False)
    category_profit.to_csv(REPORTS_DIR / "profit_by_category.csv")

# =========================
# 10. REGION / SEGMENT ANALYSIS
# =========================
print("\n========== REGION & SEGMENT ANALYSIS ==========")
for group_col in ["Region", "Segment", "State", "City"]:
    if group_col in df.columns and "Sales" in df.columns:
        grouped = df.groupby(group_col, dropna=False)["Sales"].sum().sort_values(ascending=False)
        grouped.to_csv(REPORTS_DIR / f"sales_by_{group_col.lower().replace(' ', '_')}.csv")
        print(f"\nSales by {group_col} (top 10):\n", grouped.head(10))

# =========================
# 11. MONTHLY SALES TREND
# =========================
print("\n========== MONTHLY SALES TREND ==========")
if {"Order Date", "Sales"}.issubset(df.columns):
    monthly_sales = (
        df.dropna(subset=["Order Date"])
          .set_index("Order Date")
          .resample("MS")["Sales"]
          .sum()
    )
    print(monthly_sales)
    monthly_sales.to_csv(REPORTS_DIR / "monthly_sales.csv")

    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind="line", marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "monthly_sales_trend.png", dpi=150)
    plt.close()

# =========================
# 12. FEATURE CREATION
# =========================
print("\n========== FEATURE CREATION ==========")
if "Order Date" in df.columns:
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)

if {"Sales", "Quantity"}.issubset(df.columns):
    df["Sales per Unit"] = np.where(df["Quantity"] != 0, df["Sales"] / df["Quantity"], np.nan)

if {"Profit", "Sales"}.issubset(df.columns):
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)

if {"Order Date", "Ship Date"}.issubset(df.columns):
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

# =========================
# 13. CORRELATION (NUMERIC COLUMNS)
# =========================
print("\n========== CORRELATION ==========")
numeric_df = df.select_dtypes(include=np.number)
if not numeric_df.empty:
    correlation = numeric_df.corr()
    print(correlation)
    correlation.to_csv(REPORTS_DIR / "numeric_correlation.csv")

# =========================
# 14. SAVE CLEANED DATA
# =========================
print("\n========== EXPORT ==========")
cleaned_file = PROCESSED_DIR / "superstore_cleaned.csv"
df.to_csv(cleaned_file, index=False)
print(f"Cleaned dataset saved: {cleaned_file}")

print("\n========== SCRIPT COMPLETE ==========")
print("Reports folder:", REPORTS_DIR)
print("Charts folder:", CHARTS_DIR)
