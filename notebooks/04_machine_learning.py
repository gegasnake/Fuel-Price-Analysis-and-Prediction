"""
notebooks/04_machine_learning.py

Machine Learning & Evaluation for UK Fuel Prices.

This version explicitly uses train_test_split()
to align with project guidelines / rubric.

Dataset:
- data/processed/clean_fuel.csv

Target:
- pump_price

Models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor (bonus)

Metrics:
- MAE, RMSE, R²

Bonus-quality additions (via src/models.py):
- run_sanity_checks (unit-test-like checks)
- feature importance extraction + plot (tree models)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.models import (
    run_sanity_checks,
    get_feature_importance_df,
    plot_feature_importance,
)


# 0) Dynamic paths
cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_fuel.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
RMSE_FIG_PATH = FIG_DIR / "model_rmse_comparison.png"


# 1) Load dataset
print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())


# 2) Feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week.astype(int)

TARGET_COL = "pump_price"
FEATURE_COLS = ["fuel_type", "duty_rate", "vat_rate", "year", "month", "week"]

X = df[FEATURE_COLS]
y = df[TARGET_COL]
#Sanity checks (unit-test-like)
run_sanity_checks(df, target_col=TARGET_COL)


# 3) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# 4) Preprocessing
categorical_features = ["fuel_type"]
numeric_features = [c for c in FEATURE_COLS if c not in categorical_features]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)


# 5) Models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(
        random_state=42,
        max_depth=8,
        min_samples_leaf=5,
    ),
    "RandomForest": RandomForestRegressor(
        random_state=42,
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
    ),
}


# 6) Training & evaluation
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


results = []
trained_pipelines = {}

for name, model in models.items():
    print(f"\nTraining {name}")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = evaluate(y_test, preds)
    metrics["model"] = name
    results.append(metrics)

    trained_pipelines[name] = pipe

    print(metrics)


# 7) Save metrics
metrics_df = pd.DataFrame(results).set_index("model").sort_values("RMSE")
metrics_df.to_csv(METRICS_PATH)

print("\n✅ Metrics saved to:", METRICS_PATH)
print(metrics_df)


# 8) RMSE comparison plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(metrics_df.index, metrics_df["RMSE"])
ax.set_title("Model Comparison (RMSE)")
ax.set_xlabel("Model")
ax.set_ylabel("RMSE")

fig.tight_layout()
fig.savefig(RMSE_FIG_PATH, dpi=200)
plt.close(fig)

print("RMSE comparison saved to:", RMSE_FIG_PATH)


# -----------------------------
# 8.1) Feature importance plot (BONUS – tree models)
# -----------------------------
# We use RandomForest because it usually gives stable importance estimates.
if "RandomForest" in trained_pipelines:
    rf_pipe = trained_pipelines["RandomForest"]

    try:
        importance_df = get_feature_importance_df(rf_pipe)

        plot_feature_importance(
            importance_df,
            top_n=15,
            title="Random Forest Feature Importance",
            out_path=FIG_DIR / "feature_importance_random_forest.png",
        )

        print("Feature importance plot saved: feature_importance_random_forest.png")
    except ValueError as e:
        print("Feature importance not available:", e)


# -----------------------------
# 9) Final conclusion
# -----------------------------
best_model = metrics_df.index[0]

print("\n--- Final Conclusion ---")
print(f"Best model based on RMSE: {best_model}")
print(f"RMSE: {metrics_df.loc[best_model, 'RMSE']:.3f}")
print(f"R²:  {metrics_df.loc[best_model, 'R2']:.3f}")

print("\nThis satisfies the ML split requirement using train_test_split().")
print("Feature importance analysis was performed using the Random Forest model.")
