"""
notebooks/01_data_exploration.py

Initial exploration of the RAW UK weekly fuel prices dataset (2003–present).

Exploration only:
- No cleaning or transformations are finalized here.
- We just inspect structure, types, missing values, duplicates, basic stats, and time coverage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Project root = parent of "notebooks" folder (works if this file is in notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# 1) Load raw data
# -----------------------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fuel_price.csv"

print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# 2) Basic structure
# -----------------------------
print("\nDataset shape (rows, columns):", df.shape)

print("\nColumn names:")
print(list(df.columns))


# -----------------------------
# 3) Data types and memory info
# -----------------------------
print("\nDataFrame info:")
df.info()


# -----------------------------
# 4) Missing values
# -----------------------------
print("\nMissing values per column:")
missing_counts = df.isna().sum().sort_values(ascending=False)
print(missing_counts)

missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
print("\nMissing values percentage (%):")
print(missing_percent.round(2))


# -----------------------------
# 5) Duplicates
# -----------------------------
dup_count = df.duplicated().sum()
print("\nNumber of duplicated rows:", dup_count)

if dup_count > 0:
    print("\nExample duplicated rows (first 5):")
    print(df[df.duplicated()].head())


# -----------------------------
# 6) Basic descriptive statistics (numeric)
# -----------------------------
print("\nDescriptive statistics for numeric columns:")
print(df.describe(include=[np.number]))


# -----------------------------
# 7) Check date column (raw)
# -----------------------------
# Try to guess the date column name (common in this dataset: "Date")
date_candidates = [c for c in df.columns if c.lower().strip() in ("date", "week", "week_ending", "weekending")]
date_col = date_candidates[0] if date_candidates else df.columns[0]  # fallback: first column

print(f"\nAssumed date column: {date_col}")

# Parse to datetime just for exploration (we will do final parsing in preprocessing)
df_dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

invalid_dates = df_dates.isna().sum()
print("Invalid/unparsed dates:", invalid_dates)

if invalid_dates < len(df_dates):
    print("Date range (parsed):", df_dates.min(), "to", df_dates.max())


# -----------------------------
# 8) Quick sanity checks for price columns
# -----------------------------
# Typical columns: ULSP, ULSD (case-insensitive)
cols_lower = {c.lower().strip(): c for c in df.columns}

ulsp_col = cols_lower.get("ulsp")
ulsd_col = cols_lower.get("ulsd")

price_cols = [c for c in [ulsp_col, ulsd_col] if c is not None]

print("\nDetected fuel price columns:", price_cols)

for col in price_cols:
    # Convert to numeric for exploration only
    s = pd.to_numeric(df[col], errors="coerce")
    print(f"\nColumn: {col}")
    print("  Non-null count:", s.notna().sum())
    print("  Min:", s.min())
    print("  Max:", s.max())
    print("  Mean:", s.mean())
    print("  Median:", s.median())


# -----------------------------
# 9) Quick visualizations (lightweight)
# -----------------------------
# These are simple and acceptable in exploration.
# Heavy EDA & more visuals will be done in 03_eda_visualization.ipynb

# 9.1 Histogram of ULSP and ULSD (if present)
for col in price_cols:
    plt.figure()
    s = pd.to_numeric(df[col], errors="coerce")
    plt.hist(s.dropna(), bins=40)
    plt.title(f"Distribution of {col} (raw)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 9.2 Time series plot (if we can parse dates and have price cols)
if price_cols and invalid_dates < len(df_dates):
    temp = df.copy()
    temp["_date_parsed"] = df_dates

    # Sort by date for nicer plot
    temp = temp.sort_values("_date_parsed")

    plt.figure()
    for col in price_cols:
        plt.plot(temp["_date_parsed"], pd.to_numeric(temp[col], errors="coerce"), label=col)
    plt.title("UK Weekly Fuel Prices Over Time (raw)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# -----------------------------
# 10) Summary observations (printed)
# -----------------------------
print("\n--- Summary (Exploration) ---")
print("1) This dataset contains weekly fuel prices for the UK.")
print("2) Key expected columns: Date, ULSP (petrol), ULSD (diesel).")
print("3) Next step (02_data_preprocessing) will:")
print("   - standardize column names")
print("   - parse date reliably")
print("   - reshape wide -> long (fuel_type, price)")
print("   - handle missing values, duplicates, and outliers")
print("   - save clean dataset to data/processed/clean_fuel.csv")
