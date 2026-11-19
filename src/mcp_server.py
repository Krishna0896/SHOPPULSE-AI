
from pathlib import Path
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("SHOPPULSE AI")

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

FORECAST_PATH = (
    PROJECT_DIR / "reports" / "category_sales_forecast.csv"
)


@mcp.tool()
def get_sales_forecast(category: str = "all") -> str:
    """
    Retrieve the 12-month sales forecast.

    Provide a category name:
    Furniture, Office Supplies, Technology,
    or all.
    """

    df = pd.read_csv(FORECAST_PATH)

    if category.lower() != "all":
        df = df[
            df["Category"].str.lower() == category.lower()
        ]

    if df.empty:
        return f"No forecast data found for: {category}"

    summary = (
        df.groupby("Category")["Predicted Sales"]
        .sum()
        .reset_index()
    )

    return summary.to_string(index=False)


EVALUATION_PATH = (
    PROJECT_DIR / "reports" / "forecast_evaluation.csv"
)


@mcp.tool()
def get_forecast_accuracy() -> str:
    """
    Retrieve Holt-Winters forecasting validation metrics
    for each product category.
    """

    df = pd.read_csv(EVALUATION_PATH)

    return df.to_string(index=False)


BUSINESS_ANALYSIS_PATH = (
    PROJECT_DIR / "reports" / "business_impact_analysis.csv"
)


@mcp.tool()
def get_business_analysis(dimension: str = "all") -> str:
    """
    Retrieve business performance analysis.

    Supported dimensions:
    Category, Sub-Category, Region, Segment, or all.
    """

    df = pd.read_csv(BUSINESS_ANALYSIS_PATH)

    if dimension.lower() != "all":
        df = df[
            df["Dimension"].str.lower() == dimension.lower()
        ]

    if df.empty:
        return f"No business analysis found for: {dimension}"

    return df.to_string(index=False)


@mcp.tool()
def get_recommendations() -> str:
    """
    Generate business recommendations using
    the business impact analysis results.
    """

    df = pd.read_csv(BUSINESS_ANALYSIS_PATH)

    recommendations = []

    # Identify loss-making subcategories
    subcategories = df[
        df["Dimension"] == "Sub-Category"
    ]

    loss_making = subcategories[
        subcategories["Total_Profit"] < 0
    ]

    for _, row in loss_making.iterrows():
        recommendations.append(
            f"{row['Dimension_Value']}: "
            f"Loss of ${row['Total_Profit']:,.2f}. "
            f"Review pricing, discounts and costs."
        )

    # Identify lowest-profit region
    regions = df[df["Dimension"] == "Region"]

    if not regions.empty:
        lowest_region = regions.loc[
            regions["Profit_Margin"].idxmin()
        ]

        recommendations.append(
            f"{lowest_region['Dimension_Value']} region "
            f"has the lowest profit margin of "
            f"{lowest_region['Profit_Margin']:.2%}. "
            f"Review regional discounting and profitability."
        )

    return "\n".join(recommendations)


if __name__ == "__main__":
    mcp.run()
