# SHOPPULSE AI
### AI-Powered Sales Analytics, Forecasting & Business Intelligence

SHOPPULSE AI is an end-to-end analytics and AI-assisted business intelligence project that combines Python, Machine Learning, Time-Series Forecasting, Power BI and Claude Desktop through the Model Context Protocol (MCP).

The platform transforms historical retail sales data into business insights, sales predictions and natural-language analytical assistance.

## Project Objectives

- Analyze historical retail sales and profitability.
- Build an interactive Power BI business intelligence dashboard.
- Predict transaction-level sales using Linear Regression.
- Forecast monthly category-level sales for the next 12 months using Holt-Winters Exponential Smoothing.
- Evaluate model performance using appropriate validation metrics.
- Enable Claude Desktop to access Python analytics and Power BI through MCP.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and analytics |
| Pandas, NumPy | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Linear Regression |
| Statsmodels | Holt-Winters forecasting |
| Power BI | Interactive dashboards and DAX |
| Claude Desktop | AI-assisted business analysis |
| MCP | Integration between Claude, Python and Power BI |

## Project Architecture

1. Historical sales data is processed using Python and Pandas.
2. Business metrics and analytical datasets are generated.
3. Machine Learning predicts transaction-level sales.
4. Holt-Winters forecasts monthly sales by product category.
5. Power BI presents historical performance, profitability and forecast insights.
6. Python MCP exposes forecasting and business-analysis functions to Claude Desktop.
7. Power BI Modeling MCP provides Claude access to the Power BI semantic model.

## Machine Learning

**Transaction-level Sales Prediction**

- Algorithm: Linear Regression
- MAE: 226.38
- RMSE: 693.39
- R²: 0.1861

The model outperformed a mean-prediction baseline on the test set, but its explanatory power remains limited.

## Time-Series Forecasting

**Category-level Monthly Sales Forecast**

- Algorithm: Holt-Winters Exponential Smoothing
- Trend: Additive
- Seasonality: Additive
- Seasonal period: 12 months
- Forecast horizon: 12 months

Forecast validation used 2014–2016 for training and 2017 for testing.

| Category | Validation MAE | Validation RMSE |
|---|---:|---:|
| Furniture | 3,979.85 | 4,574.42 |
| Office Supplies | 5,595.51 | 7,603.40 |
| Technology | 8,603.94 | 10,507.78 |

The forecasts are exploratory and should be interpreted with caution because the historical dataset contains only 48 monthly observations per category.

## Claude Desktop and MCP Integration

Four custom Python MCP tools were implemented:

- `get_sales_forecast`
- `get_forecast_accuracy`
- `get_business_analysis`
- `get_recommendations`

The project also integrates Microsoft's Power BI Modeling MCP server.

This allows Claude Desktop to retrieve Python analytics and query the Power BI semantic model through connected MCP tools.

## Business Insights

The analysis identifies:

- Loss-making subcategories such as Tables, Bookcases and Supplies.
- Regional differences in profitability.
- Category-level sales forecasts.
- Areas requiring further investigation into pricing, discounts and profitability.

Recommendations are based on observed historical associations and should not be interpreted as proof of causation.

## Repository Structure

```text
SHOPPULSE_AI/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── charts/
├── src/
│   ├── analyze_business_impact.py
│   ├── create_historical_forecast.py
│   ├── evaluate_forecast.py
│   ├── forecast_sales.py
│   ├── generate_recommendations.py
│   ├── mcp_server.py
│   ├── prepare_forecasting_data.py
│   └── prepare_ml_data.py
└── .gitignore
```

## Future Scope

- Improve model validation using additional historical observations.
- Enhance AI-assisted business recommendations.
- Expand analytical capabilities through additional MCP tools.

## Disclaimer

This project is developed for learning, portfolio demonstration and business analytics experimentation. Forecasts and model predictions are estimates, not guaranteed outcomes.
