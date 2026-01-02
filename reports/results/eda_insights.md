# EDA Insights – UK Weekly Fuel Prices

## Dataset
- Source: UK weekly fuel price statistics (2003–present)
- Processed file used: `data/processed/clean_fuel.csv`
- Date range: 2003-06-09 to 2020-11-02
- Rows (after long reshape): 1818

## Key Findings (from EDA)
1. **Diesel vs Petrol:** On average, diesel differs from petrol by approximately **4.54 pence/litre** (diesel - petrol).
2. **Long-term Trend:** Prices change substantially over time (see `price_trend_over_time.png`), indicating strong temporal effects.
3. **Duty and VAT Characteristics:**
   - Duty rate variability (std) ≈ **4.6635**
   - VAT rate variability (std) ≈ **1.5322**
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
