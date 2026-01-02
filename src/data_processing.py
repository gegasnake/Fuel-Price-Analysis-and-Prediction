# src/data_processing.py
"""
Reusable data-loading and preprocessing utilities for the project.

Dataset: UK weekly fuel prices (2003–present)
- ULSP = Ultra low sulphur unleaded petrol
- ULSD = Ultra low sulphur diesel

CSV has these columns BEFORE cleaning:
- Unnamed: 0
- Date
- Pump price in pence/litre (ULSP)
- Pump price in pence/litre (ULSD)
- Duty rate in pence/litre (ULSP)
- Duty rate in pence/litre (ULSD)
- VAT percentage rate (ULSP)
- VAT percentage rate (ULSD)

Goal of this module (Collaborator Gega):
- Load raw CSV from data/raw/
- Standardize column names
- Parse date column
- Reshape data from wide format into long format:
    date | fuel_type | pump_price | duty_rate | vat_rate
- Clean data: missing values, duplicates, optional outliers
"""

from __future__ import annotations

from typing import Iterable, Optional
import pandas as pd


# -----------------------------
# Loading
# -----------------------------
def load_data(path: str, *, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    return pd.read_csv(path, encoding=encoding)


# -----------------------------
# Basic standardization
# -----------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - lowercase
    - strip spaces
    - replace spaces with underscores

    NOTE:
    This does NOT remove special characters like ':' or '/'.
    Your dataset will become columns like:
      'pump_price_in_pence/litre_(ulsp)'
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def parse_date_column(
    df: pd.DataFrame,
    date_col: str = "date",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Convert a date column to pandas datetime.
    - dayfirst=True works well for typical UK DD/MM/YYYY formats.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    return df


# -----------------------------
# Reshaping (UK fuel-specific for YOUR CSV)
# -----------------------------
def reshape_uk_fuel_prices(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Reshape the UK dataset from wide columns into long format, preserving pump price,
    duty and VAT as separate columns.

    Expected columns (AFTER clean_column_names):
      - pump_price_in_pence/litre_(ulsp)
      - pump_price_in_pence/litre_(ulsd)
      - duty_rate_in_pence/litre_(ulsp)
      - duty_rate_in_pence/litre_(ulsd)
      - vat_percentage_rate_(ulsp)
      - vat_percentage_rate_(ulsd)

    Output schema:
      date | fuel_type | pump_price | duty_rate | vat_rate
    """
    df = df.copy()

    # Your actual column names after clean_column_names()
    pump_ulsp = "pump_price_in_pence/litre_(ulsp)"
    pump_ulsd = "pump_price_in_pence/litre_(ulsd)"
    duty_ulsp = "duty_rate_in_pence/litre_(ulsp)"
    duty_ulsd = "duty_rate_in_pence/litre_(ulsd)"
    vat_ulsp = "vat_percentage_rate_(ulsp)"
    vat_ulsd = "vat_percentage_rate_(ulsd)"

    required = {date_col, pump_ulsp, pump_ulsd, duty_ulsp, duty_ulsd, vat_ulsp, vat_ulsd}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Build petrol rows
    petrol = df[[date_col, pump_ulsp, duty_ulsp, vat_ulsp]].copy()
    petrol.columns = [date_col, "pump_price", "duty_rate", "vat_rate"]
    petrol["fuel_type"] = "petrol"

    # Build diesel rows
    diesel = df[[date_col, pump_ulsd, duty_ulsd, vat_ulsd]].copy()
    diesel.columns = [date_col, "pump_price", "duty_rate", "vat_rate"]
    diesel["fuel_type"] = "diesel"

    out = pd.concat([petrol, diesel], ignore_index=True)

    # Reorder for a clean contract
    out = out[[date_col, "fuel_type", "pump_price", "duty_rate", "vat_rate"]]

    return out


# -----------------------------
# Cleaning utilities
# -----------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows."""
    return df.drop_duplicates().copy()


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    *,
    target_numeric_cols: Optional[Iterable[str]] = None,
    target_categorical_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Fill missing values:
    - numeric columns: median (default) or mean
    - categorical columns: mode
    """
    df = df.copy()

    # Numeric columns to fill
    num_cols = df.select_dtypes(include="number").columns if target_numeric_cols is None else list(target_numeric_cols)
    for col in num_cols:
        if df[col].isna().any():
            if numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            else:
                raise ValueError("numeric_strategy must be 'median' or 'mean'")

    # Categorical columns to fill
    cat_cols = (
        df.select_dtypes(include=["object", "category"]).columns
        if target_categorical_cols is None
        else list(target_categorical_cols)
    )
    for col in cat_cols:
        if df[col].isna().any():
            if categorical_strategy == "mode":
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
            else:
                raise ValueError("categorical_strategy must be 'mode'")

    return df


def drop_invalid_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Drop rows where date parsing failed (date is NaT)."""
    df = df.copy()
    return df[df[date_col].notna()]


def handle_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR rule:
      lower = Q1 - k*IQR
      upper = Q3 + k*IQR
    """
    df = df.copy()
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df

    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


# -----------------------------
# Full pipeline (one function)
# -----------------------------
def build_clean_dataset(
    raw_csv_path: str,
    *,
    encoding: Optional[str] = None,
    dayfirst: bool = True,
    remove_outliers: bool = False,
    outlier_k: float = 1.5,
) -> pd.DataFrame:
    """
    End-to-end pipeline for YOUR dataset.

    Output schema:
      date (datetime64)
      fuel_type (petrol/diesel)
      pump_price (float)   <- target
      duty_rate (float)
      vat_rate (float)
    """
    df = load_data(raw_csv_path, encoding=encoding)
    df = clean_column_names(df)

    # Drop index-like column if present (common: 'Unnamed: 0' -> 'unnamed:_0')
    if "unnamed:_0" in df.columns:
        df = df.drop(columns=["unnamed:_0"])

    df = parse_date_column(df, date_col="date", dayfirst=dayfirst)
    df = drop_invalid_dates(df, date_col="date")

    # Wide -> long, keep pump_price + duty + vat
    df = reshape_uk_fuel_prices(df, date_col="date")

    # Ensure numeric types
    for col in ["pump_price", "duty_rate", "vat_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fuel_type"] = df["fuel_type"].astype("category")

    # Fill missing values
    df = handle_missing_values(df, numeric_strategy="median", categorical_strategy="mode")

    # Remove duplicates
    df = remove_duplicates(df)

    # Optional outlier removal (typically apply to target: pump_price)
    if remove_outliers:
        df = handle_outliers_iqr(df, column="pump_price", k=outlier_k)

    # Sort for nicer downstream time-series plots
    df = df.sort_values(["date", "fuel_type"]).reset_index(drop=True)

    return df
