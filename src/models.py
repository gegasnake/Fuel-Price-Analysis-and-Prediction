"""
src/models.py

Reusable model utilities for UK Fuel Price Prediction.

PUBLIC API (main functions you are expected to use):
- get_feature_columns() -> dict of feature lists
- build_preprocessor() -> ColumnTransformer
- get_models() -> dict of model instances
- train_model(model, X_train, y_train) -> trained Pipeline
- evaluate_model(pipeline, X_test, y_test) -> dict of metrics
- get_feature_importance_df(pipeline, preprocessor) -> DataFrame (tree models only)
- plot_feature_importance(importance_df, out_path=...) -> saves a bar chart
- run_sanity_checks(df, target_col=...) -> unit-test-like checks for dataset/pipeline

NOTES
- This module does not load data from disk (keeps it reusable).
- Visualization uses matplotlib only (course-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Feature configuration (single source of truth)
@dataclass(frozen=True)
class FeatureConfig:
    """
    Defines which columns the project uses as features.
    """
    categorical: List[str]
    numeric: List[str]

    @property
    def all_features(self) -> List[str]:
        return self.categorical + self.numeric


def get_feature_columns() -> FeatureConfig:
    """
    Return the standard feature configuration used across the project.

    Expected input DataFrame columns:
    - fuel_type (petrol/diesel)
    - duty_rate, vat_rate (numeric)
    - year, month, week (engineered numeric time features)
    """
    return FeatureConfig(
        categorical=["fuel_type"],
        numeric=["duty_rate", "vat_rate", "year", "month", "week"],
    )


# Preprocessing
def build_preprocessor(feature_cfg: Optional[FeatureConfig] = None) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - OneHotEncode categorical features
    - Pass-through numeric features

    Returns
    -------
    ColumnTransformer
    """
    cfg = feature_cfg or get_feature_columns()

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.categorical),
            ("num", "passthrough", cfg.numeric),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get final feature names after preprocessing (including one-hot encoded names).

    This is needed for feature importance and interpretability.
    """
    # After fitting, the OneHotEncoder exposes get_feature_names_out
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out()

    # Numeric passthrough retains original names
    # We stored numeric feature names in the transformer config
    # But ColumnTransformer doesn't always expose them cleanly, so we read from transformers_
    numeric_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            numeric_features = list(cols)

    return list(cat_features) + numeric_features


# Model factory
def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Return a dictionary of regression models used in the project.
    """
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            random_state=random_state,
            max_depth=8,
            min_samples_leaf=5,
        ),
        "RandomForest": RandomForestRegressor(
            random_state=random_state,
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    }


# Training
def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    feature_cfg: Optional[FeatureConfig] = None,
) -> Pipeline:
    """
    Train a model with preprocessing pipeline.

    Parameters
    ----------
    model : sklearn estimator
    X_train : pd.DataFrame
    y_train : pd.Series
    feature_cfg : FeatureConfig, optional

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline with steps:
        - preprocess
        - model
    """
    preprocess = build_preprocessor(feature_cfg)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


# Evaluation
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate regression model and return metrics (MAE, RMSE, R²).
    """
    preds = pipeline.predict(X_test)

    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(mean_squared_error(y_test, preds) ** 0.5),
        "R2": float(r2_score(y_test, preds)),
    }



# Feature importance (tree models) + plots
def get_feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract feature importance for tree-based models in the pipeline.

    Returns
    -------
    pd.DataFrame with columns:
    - feature
    - importance

    Raises
    ------
    ValueError if model does not support feature_importances_
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importance (no feature_importances_ attribute).")

    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) != importance count ({len(importances)})."
        )

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Feature Importance (Top Features)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot feature importance as a bar chart (matplotlib only).

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output from get_feature_importance_df()
    top_n : int
        How many top features to display
    out_path : Optional[Path]
        If provided, saves figure to this path

    Returns
    -------
    matplotlib.figure.Figure
    """
    d = importance_df.head(top_n).copy()
    d = d.iloc[::-1]  # reverse so highest is on top in horizontal bar chart

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.barh(d["feature"], d["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return fig



# Unit-test-like checks (sanity checks)
def run_sanity_checks(
    df: pd.DataFrame,
    *,
    target_col: str = "pump_price",
    feature_cfg: Optional[FeatureConfig] = None,
) -> None:
    """
    Lightweight unit-test-like checks for data and pipeline assumptions.
    Raise clear errors if something is wrong.

    Use this in ML notebook before training to catch issues early.
    """
    cfg = feature_cfg or get_feature_columns()

    required_cols = set(cfg.all_features + [target_col])
    missing = required_cols - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns in dataset: {missing}")

    # Check types / parseability
    if "fuel_type" in df.columns:
        allowed = {"petrol", "diesel"}
        observed = set(df["fuel_type"].dropna().unique())
        if not observed.issubset(allowed):
            raise AssertionError(f"Unexpected fuel_type values: {observed - allowed}")

    # Numeric columns should be numeric or convertible
    for col in cfg.numeric + [target_col]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try converting (do not modify df, just check)
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.isna().all():
                raise AssertionError(f"Column '{col}' is not numeric and cannot be converted.")

    # No all-null columns
    for col in cfg.all_features + [target_col]:
        if df[col].isna().all():
            raise AssertionError(f"Column '{col}' is entirely null.")

    print("✅ Sanity checks passed: dataset has required columns and valid basic constraints.")


def quick_smoke_train(
    df: pd.DataFrame,
    *,
    target_col: str = "pump_price",
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Optional quick smoke-test:
    - Trains all models on a small subset
    - Returns metrics dict
    Useful to ensure everything runs end-to-end without full notebook execution.
    """
    from sklearn.model_selection import train_test_split

    cfg = get_feature_columns()
    run_sanity_checks(df, target_col=target_col, feature_cfg=cfg)

    X = df[cfg.all_features]
    y = df[target_col]

    # Small subset to keep it fast
    if len(df) > 5000:
        df_small = df.sample(n=5000, random_state=random_state)
        X = df_small[cfg.all_features]
        y = df_small[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    results: Dict[str, Dict[str, float]] = {}
    for name, model in get_models(random_state=random_state).items():
        pipe = train_model(model, X_train, y_train, feature_cfg=cfg)
        results[name] = evaluate_model(pipe, X_test, y_test)

    print("Smoke test complete.")
    return results
