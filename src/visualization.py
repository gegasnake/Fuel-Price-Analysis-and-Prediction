# src/visualization.py
"""
src/visualization.py

Reusable visualization utilities for the project.
Collaborator B owns this file.

Input dataset (cleaned):
data/processed/clean_fuel.csv

Expected columns:
- date (datetime-like or string parseable to datetime)
- fuel_type (petrol/diesel)
- pump_price (target)
- duty_rate
- vat_rate

All plotting functions here:
- create a single figure
- label axes + title
- optionally save to reports/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 200) -> None:
    """Save a matplotlib figure with tight layout."""
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_price_trend_over_time(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "pump_price",
    fuel_col: str = "fuel_type",
    title: str = "Pump Price Over Time (Petrol vs Diesel)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Line plot of pump price over time with one line per fuel type.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    # Sort for clean line plots
    d = d.sort_values([fuel_col, date_col])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for fuel, grp in d.groupby(fuel_col):
        ax.plot(grp[date_col], grp[price_col], label=str(fuel))

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Pump Price (pence/litre)")
    ax.legend()

    if out_path is not None:
        save_fig(fig, out_path)

    return fig


def plot_price_distribution_by_fuel(
    df: pd.DataFrame,
    *,
    price_col: str = "pump_price",
    fuel_col: str = "fuel_type",
    title: str = "Distribution of Pump Price by Fuel Type",
    out_path: Optional[Path] = None,
    bins: int = 40,
) -> plt.Figure:
    """
    Histogram distribution per fuel type on the same axes.
    """
    d = df.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for fuel, grp in d.groupby(fuel_col):
        ax.hist(grp[price_col].dropna(), bins=bins, alpha=0.5, label=str(fuel))

    ax.set_title(title)
    ax.set_xlabel("Pump Price (pence/litre)")
    ax.set_ylabel("Frequency")
    ax.legend()

    if out_path is not None:
        save_fig(fig, out_path)

    return fig


def plot_boxplot_price_by_fuel(
    df: pd.DataFrame,
    *,
    price_col: str = "pump_price",
    fuel_col: str = "fuel_type",
    title: str = "Pump Price by Fuel Type (Boxplot)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Boxplot comparing pump price across fuel types.
    """
    d = df.copy()
    fuels = list(d[fuel_col].dropna().unique())
    data = [d.loc[d[fuel_col] == f, price_col].dropna().values for f in fuels]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot(data, labels=[str(f) for f in fuels], showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Pump Price (pence/litre)")

    if out_path is not None:
        save_fig(fig, out_path)

    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot (x vs y). Kept simple (no seaborn) for course clarity.
    """
    d = df[[x_col, y_col]].copy()
    d = d.dropna()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(d[x_col], d[y_col], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)

    if out_path is not None:
        save_fig(fig, out_path)

    return fig


def plot_monthly_average_price(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "pump_price",
    fuel_col: str = "fuel_type",
    title: str = "Average Monthly Pump Price (Seasonality)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot average pump price by month (1..12) for each fuel type to show seasonality.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["month"] = d[date_col].dt.month

    monthly = (
        d.groupby([fuel_col, "month"], as_index=False)[price_col]
        .mean()
        .sort_values(["month"])
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for fuel, grp in monthly.groupby(fuel_col):
        ax.plot(grp["month"], grp[price_col], marker="o", label=str(fuel))

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Pump Price (pence/litre)")
    ax.set_xticks(range(1, 13))
    ax.legend()

    if out_path is not None:
        save_fig(fig, out_path)

    return fig

def plot_interactive_price_trend(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "pump_price",
    fuel_col: str = "fuel_type",
    title: str = "Interactive Pump Price Trend (Petrol vs Diesel)",
    out_path: Optional[Path] = None,
):
    """
    Interactive Plotly line chart for pump price over time.

    Saved as HTML so it can be opened without a Python environment.
    (Bonus: Advanced Visualization)
    """
    import plotly.express as px

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    fig = px.line(
        d.sort_values(date_col),
        x=date_col,
        y=price_col,
        color=fuel_col,
        title=title,
        labels={
            date_col: "Date",
            price_col: "Pump Price (pence/litre)",
            fuel_col: "Fuel Type",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Fuel Type",
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path)

    return fig
