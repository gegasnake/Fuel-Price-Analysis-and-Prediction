# notebooks/02_data_preprocessing.py
"""
Data preprocessing pipeline for the UK Weekly Fuel Prices dataset (2003–present).

Portable script:
- Uses src/data_processing.py utilities.

Output (contract file for the team):
- data/processed/clean_fuel.csv

Final columns (long format):
- date
- fuel_type (petrol/diesel)
- pump_price (target)
- duty_rate
- vat_rate
"""

from pathlib import Path
import pandas as pd

from src.data_processing import (
    load_data,
    clean_column_names,
    parse_date_column,
    reshape_uk_fuel_prices,
    handle_missing_values,
    remove_duplicates,
    handle_outliers_iqr,
    drop_invalid_dates,
)

# -----------------------------
# 0) Dynamic, team-safe paths
# -----------------------------
cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fuel_price.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_fuel.csv"


# -----------------------------
# 1) Load raw data
# -----------------------------
print("Loading raw dataset...")
df_raw = load_data(str(RAW_DATA_PATH))

print("\nRaw dataset preview:")
print(df_raw.head())


# -----------------------------
# 2) Initial data quality (BEFORE cleaning)
# -----------------------------
print("\n--- Initial Data Quality (BEFORE cleaning) ---")
print("Shape (rows, cols):", df_raw.shape)

print("\nData types and non-null counts:")
df_raw.info()

print("\nMissing values per column:")
print(df_raw.isna().sum())

print("\nDuplicate rows:", df_raw.duplicated().sum())


# -----------------------------
# 3) Standardize column names
# -----------------------------
df = clean_column_names(df_raw)
print("\nStandardized column names:")
print(list(df.columns))

# Drop index-like column if present (your file has it)
if "unnamed:_0" in df.columns:
    df = df.drop(columns=["unnamed:_0"])
    print("\nDropped index-like column: 'unnamed:_0'")


# -----------------------------
# 4) Parse date column
# -----------------------------
df = parse_date_column(df, date_col="date", dayfirst=True)
df = drop_invalid_dates(df, date_col="date")


# -----------------------------
# 5) Reshape wide -> long
#    Output: date | fuel_type | pump_price | duty_rate | vat_rate
# -----------------------------
df = reshape_uk_fuel_prices(df, date_col="date")

# Ensure correct dtypes (safe even if already numeric)
for col in ["pump_price", "duty_rate", "vat_rate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["fuel_type"] = df["fuel_type"].astype("category")


# -----------------------------
# 6) Handle missing values
# -----------------------------
df = handle_missing_values(
    df,
    numeric_strategy="median",
    categorical_strategy="mode",
    target_numeric_cols=["pump_price", "duty_rate", "vat_rate"],  # explicit for clarity
    target_categorical_cols=["fuel_type"],
)


# -----------------------------
# 7) Remove duplicates
# -----------------------------
df = remove_duplicates(df)


# -----------------------------
# 8) Outlier handling (IQR) - apply to target pump_price
# -----------------------------
REMOVE_OUTLIERS = True
if REMOVE_OUTLIERS:
    before = len(df)
    df = handle_outliers_iqr(df, column="pump_price", k=1.5)
    after = len(df)
    print(f"\nOutlier removal enabled: removed {before - after} rows using IQR on pump_price.")


# -----------------------------
# 9) Final checks (AFTER cleaning)
# -----------------------------
df = df.sort_values(["date", "fuel_type"]).reset_index(drop=True)

print("\n--- Final Data Quality (AFTER cleaning) ---")
print("Final shape (rows, cols):", df.shape)

print("\nFinal missing values per column:")
print(df.isna().sum())

print("\nFinal duplicate rows:", df.duplicated().sum())

print("\nFinal preview:")
print(df.head())


# -----------------------------
# 10) Save processed dataset
# -----------------------------
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\n✅ Cleaned dataset saved to: {PROCESSED_DATA_PATH}")


# -----------------------------
# 11) Output contract for team
# -----------------------------
print("\n--- Output Contract ---")
print("File:", PROCESSED_DATA_PATH)
print("Columns:", list(df.columns))
print("Expected columns: ['date', 'fuel_type', 'pump_price', 'duty_rate', 'vat_rate']")
print("Fuel types:", df["fuel_type"].unique().tolist())
