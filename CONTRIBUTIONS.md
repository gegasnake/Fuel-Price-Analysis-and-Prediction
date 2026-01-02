# Contributions

This document describes the contributions of each collaborator to the **UK Fuel Prices Analysis and Prediction** project.  
The project was developed collaboratively, with clear task ownership and separation of responsibilities.

---

## 👤 Collaborator GEGA — Data Ingestion & Preprocessing

**Responsibilities:**
- Dataset selection and validation
- Data ingestion and cleaning
- Data reshaping and preprocessing pipeline

**Key Contributions:**
- Selected and validated the UK weekly fuel prices dataset (2003–present)
- Implemented reusable data utilities in `src/data_processing.py`
- Performed initial data exploration in `01_data_exploration.ipynb`
- Cleaned and reshaped the raw dataset into ML-ready long format
- Engineered a consistent processed dataset:
data/processed/clean_fuel.csv
- Ensured reproducibility using project-relative paths
- Handled missing values, duplicates, and outliers
- Defined the final data contract used by all collaborators

**Files Owned:**
- `src/data_processing.py`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `data/processed/clean_fuel.csv`

---

## 👤 Collaborator BESO — Exploratory Data Analysis & Visualization

**Responsibilities:**
- Exploratory data analysis (EDA)
- Visual analysis and interpretation
- Reporting EDA insights

**Key Contributions:**
- Performed comprehensive EDA using the cleaned dataset
- Created multiple visualization types including:
- Time-series plots
- Distribution plots
- Boxplots
- Scatter plots
- Correlation heatmap
- Implemented reusable plotting utilities in `src/visualization.py`
- Added interactive Plotly visualization (bonus)
- Summarized insights in a dedicated EDA report

**Files Owned:**
- `src/visualization.py`
- `notebooks/03_eda_visualization.ipynb`
- `reports/figures/`
- `reports/results/eda_insights.md`

---

## 👤 Collaborator GUKA — Machine Learning & Evaluation

**Responsibilities:**
- Feature engineering
- Model training and evaluation
- Model comparison and conclusions

**Key Contributions:**
- Engineered time-based features from dates (year, month, week)
- Implemented regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor (bonus)
- Used `train_test_split()` to align with project guidelines
- Evaluated models using MAE, RMSE, and R² metrics
- Compared model performance visually and numerically
- Generated final model metrics and conclusions

**Files Owned:**
- `notebooks/04_machine_learning.ipynb`
- `reports/results/model_metrics.csv`
- `reports/figures/model_rmse_comparison.png`

---

## 👤 Collaborator NINO — Model Abstraction & Engineering Quality

**Responsibilities:**
- Code abstraction and reusability
- Model interpretability
- Engineering best practices

**Key Contributions:**
- Created `src/models.py` to abstract:
- Feature configuration
- Preprocessing pipelines
- Model creation and training
- Evaluation metrics
- Implemented feature importance extraction and plotting
- Added unit-test-like sanity checks for data and pipelines
- Improved documentation and maintainability of ML components
- Supported bonus objectives related to code quality and creativity

**Files Owned:**
- `src/models.py`

---

## 🤝 Collaboration Principles

- Each collaborator worked on **clearly defined components**
- All modules interact via **well-defined data contracts**
- No collaborator modified another’s owned files without coordination
- All code follows reproducibility and readability best practices

---

## 📌 Notes

This project emphasizes:
- Clear separation of concerns
- Reproducible data science workflows
- Explainable and interpretable machine learning
- Professional software engineering practices

All contributors participated actively and collaboratively throughout the project lifecycle.
