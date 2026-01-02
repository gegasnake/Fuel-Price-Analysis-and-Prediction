"""
notebooks/03_eda_visualization.py

Exploratory Data Analysis (EDA) + Visualization for the project.

Rules:
- Use ONLY the processed dataset: data/processed/clean_fuel.csv
- Save figures to: reports/figures/
- Write insights to: reports/results/eda_insights.md

"""

from pathlib import Path
import pandas as pd

from src.visualization import (
    plot_price_trend_over_time,
    plot_price_distribution_by_fuel,
    plot_boxplot_price_by_fuel,
    plot_scatter_with_trend,
    plot_monthly_average_price,
    plot_interactive_price_trend,
    plot_correlation_heatmap,
)
# -----------------------------
# 0) Dynamic, team-safe paths
# -----------------------------
cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_fuel.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
INSIGHTS_PATH = RESULTS_DIR / "eda_insights.md"

# Ensure output dirs exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Load cleaned data
# -----------------------------
print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

# Parse date for analysis
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

print("\nDataset shape:", df.shape)
print("Columns:", list(df.columns))
print("\nPreview:")
print(df.head())

# -----------------------------
# 2) Basic descriptive stats
# -----------------------------
print("\n--- Basic Stats ---")
print(df[["pump_price", "duty_rate", "vat_rate"]].describe())

print("\n--- Price by Fuel Type (mean/median) ---")
grp_stats = df.groupby("fuel_type")["pump_price"].agg(["count", "mean", "median", "min", "max"])
print(grp_stats)

# -----------------------------
# 3) Visualizations (saved to reports/figures/)
# -----------------------------

# 3.1 Trend over time
plot_price_trend_over_time(
    df,
    out_path=FIG_DIR / "price_trend_over_time.png",
)

# 3.2 Distribution (histograms)
plot_price_distribution_by_fuel(
    df,
    out_path=FIG_DIR / "price_distribution_by_fuel.png",
)

# 3.3 Boxplot
plot_boxplot_price_by_fuel(
    df,
    out_path=FIG_DIR / "price_boxplot_by_fuel.png",
)

# 3.4 Scatter: duty vs pump price
plot_scatter_with_trend(
    df,
    x_col="duty_rate",
    y_col="pump_price",
    title="Pump Price vs Duty Rate",
    x_label="Duty Rate (pence/litre)",
    y_label="Pump Price (pence/litre)",
    out_path=FIG_DIR / "price_vs_duty_scatter.png",
)

# 3.5 Scatter: VAT vs pump price
plot_scatter_with_trend(
    df,
    x_col="vat_rate",
    y_col="pump_price",
    title="Pump Price vs VAT Rate",
    x_label="VAT Rate (%)",
    y_label="Pump Price (pence/litre)",
    out_path=FIG_DIR / "price_vs_vat_scatter.png",
)

# 3.6 Monthly seasonality
plot_monthly_average_price(
    df,
    out_path=FIG_DIR / "monthly_avg_price_seasonality.png",
)

print(f"\n✅ Figures saved to: {FIG_DIR}")

# 4) Write EDA insights
# Diesel vs petrol difference (average)
mean_by_fuel = df.groupby("fuel_type")["pump_price"].mean()
if "diesel" in mean_by_fuel.index and "petrol" in mean_by_fuel.index:
    diff = float(mean_by_fuel["diesel"] - mean_by_fuel["petrol"])
else:
    diff = float("nan")

# Duty and VAT variability (std)
duty_std = float(df["duty_rate"].std())
vat_std = float(df["vat_rate"].std())

# Time span
start_date = df["date"].min().date()
end_date = df["date"].max().date()

# -----------------------------
# Correlation Heatmap
# -----------------------------
numeric_cols = [
    "pump_price",
    "duty_rate",
    "vat_rate",
]

plot_correlation_heatmap(
    df,
    cols=numeric_cols,
    title="Correlation Heatmap: Pump Price, Duty Rate, VAT Rate",
    out_path=FIG_DIR / "correlation_heatmap.png",
)

print("✅ Correlation heatmap saved: correlation_heatmap.png")


insights_text = f"""# EDA Insights – UK Weekly Fuel Prices

## Dataset
- Source: UK weekly fuel price statistics (2003–present)
- Processed file used: `data/processed/clean_fuel.csv`
- Date range: {start_date} to {end_date}
- Rows (after long reshape): {len(df)}

## Key Findings (from EDA)
1. **Diesel vs Petrol:** On average, diesel differs from petrol by approximately **{diff:.2f} pence/litre** (diesel - petrol).
2. **Long-term Trend:** Prices change substantially over time (see `price_trend_over_time.png`), indicating strong temporal effects.
3. **Duty and VAT Characteristics:**
   - Duty rate variability (std) ≈ **{duty_std:.4f}**
   - VAT rate variability (std) ≈ **{vat_std:.4f}**
   These indicate how stable/variable taxation-related components are relative to pump price.
4. **Distribution:** Price distributions differ by fuel type (see `price_distribution_by_fuel.png` and `price_boxplot_by_fuel.png`).
5. **Seasonality:** Monthly averages show potential seasonal patterns (see `monthly_avg_price_seasonality.png`).

## Figures Generated
- `price_trend_over_time.png`
- `price_distribution_by_fuel.png`
- `price_boxplot_by_fuel.png`
- `price_vs_duty_scatter.png`
- `price_vs_vat_scatter.png`
- `monthly_avg_price_seasonality.png`

## Notes / Next Steps
- In the ML notebook, create time-based features (year, month, week) and compare models:
  Linear Regression vs Decision Tree vs (bonus) Random Forest.
"""

INSIGHTS_PATH.write_text(insights_text, encoding="utf-8")
print(f"\n✅ EDA insights written to: {INSIGHTS_PATH}")


# BONUS: Interactive Plotly visualization
plot_interactive_price_trend(
    df,
    out_path=FIG_DIR / "interactive_price_trend.html",
)

print("Interactive Plotly chart saved: interactive_price_trend.html")