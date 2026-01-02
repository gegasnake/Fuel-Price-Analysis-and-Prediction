# UK Fuel Prices Analysis and Prediction

## 📌 Project Overview
This project analyzes **weekly UK fuel prices (2003–present)** and builds machine learning models to understand and predict pump prices based on fuel type, taxation components, and time-related features.

The project follows a reproducible data science pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, and model evaluation.

---

## 🎯 Objectives
- Analyze long-term trends in UK fuel prices
- Compare petrol and diesel price behavior
- Explore relationships between pump prices, duty rates, and VAT
- Build and evaluate regression models to predict fuel prices
- Apply clean software engineering and documentation practices

---

## 📂 Project Structure
```text
FuelStationDataAnalysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualization.ipynb
│   └── 04_machine_learning.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── visualization.py
│   └── models.py
│
├── reports/
│   ├── figures/
│   └── results/
│
├── README.md
├── CONTRIBUTIONS.md
└── requirements.txt
```
---

## 📊 Dataset
- Source: UK government weekly fuel price statistics
- Time span: 2003 – present
- Frequency: Weekly

Processed dataset:
data/processed/clean_fuel.csv

---

## 🔄 Workflow Summary
1. Data exploration and validation
2. Data preprocessing and cleaning
3. Exploratory data analysis and visualization
4. Machine learning modeling and evaluation

---

## 🤖 Models & Evaluation
Models implemented:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation metrics:
- MAE
- RMSE
- R²

---

## 🏆 Bonus Features
- Feature engineering from timestamps
- Additional ML model (Random Forest)
- Correlation heatmap
- Feature importance plots
- Unit-test-like sanity checks
- Modular and well-documented codebase

---

## ▶️ How to Run

From the project root directory:

pip install -r requirements.txt  
PYTHONPATH=. python notebooks/04_machine_learning.py

---

## 👥 Contributors
See CONTRIBUTIONS.md for detailed contributor roles.

---

## 📌 Notes
This project demonstrates a full end-to-end data science workflow with an emphasis on clarity, reproducibility, and interpretability.
