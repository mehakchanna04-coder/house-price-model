# House Price Prediction — ML Model Building & Comparison

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-R2%3D0.9066-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Project%202-orange)

---

## Project Overview

This is **Project 2** of the DecodeLabs Industrial Training Kit (Batch 2026).
Built on top of the cleaned dataset from Project 1, this project trains and
compares three machine learning models — Linear Regression, Random Forest,
and XGBoost — to predict house sale prices on the Ames Housing dataset.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Project 1 — validated_clean_train.csv |
| Shape | 1460 rows × 79 columns |
| Target Variable | SalePrice |
| Features | 78 (including 5 engineered features) |

---

## Project Structure
house-price-model/

│

├── data/

│   └── validated_clean_train.csv   ← cleaned data from Project 1

│

├── notebooks/

│   └── model_building.ipynb        ← full model training notebook

│

├── src/

│   ├── train_models.py             ← train all 3 models

│   ├── evaluate.py                 ← metrics, plots, feature importance

│   └── predict.py                  ← load model & make predictions

│

├── outputs/

│   ├── best_model_xgboost.pkl      ← saved XGBoost model

│   ├── model_comparison.csv        ← results table

│   ├── model_comparison.png        ← comparison chart

│   └── feature_importance.png      ← feature importance plot

│

└── README.md
---

## Models Trained & Compared

| Model | RMSE | MAE | R2 Score |
|-------|------|-----|----------|
| Linear Regression | $36,313 | $20,002 | 0.8281 |
| Random Forest | $27,747 | $16,956 | 0.8896 |
| XGBoost | $26,770 | $17,164 | **0.9066** |

**Winner: XGBoost** with R2 = 0.9066

XGBoost explains **90.66% of variance** in house prices.

---

## Top 5 Most Important Features

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | OverallQual | 0.315 | Original |
| 2 | total_area | 0.136 | Engineered in Project 1 |
| 3 | GarageCars | 0.074 | Original |
| 4 | KitchenQual | 0.066 | Original |
| 5 | GarageFinish | 0.058 | Original |

`total_area` — engineered in Project 1 — is the **2nd most important
feature**, proving that good feature engineering directly improves model
performance.

---

## Pipeline
Raw Data (Project 1 output)

↓

One-Hot Encoding (categorical columns)

↓

Train/Test Split (80/20)

↓

Train 3 Models simultaneously

↓

Evaluate — RMSE, MAE, R2

↓

Compare & Select Best Model

↓

Save Best Model (.pkl)
---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/mehakchanna04-coder/house-price-model.git
cd house-price-model
```

**2. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

**3. Add cleaned dataset:**
- Copy `validated_clean_train.csv` from Project 1
- Place it in the `data/` folder

**4. Train all models:**
```bash
python src/train_models.py
```

**5. Evaluate models:**
```bash
python src/evaluate.py
```

**6. Make predictions:**
```bash
python src/predict.py
```

**7. Or open the notebook:**
```bash
jupyter notebook notebooks/model_building.ipynb
```

---

## Key Learnings

- XGBoost outperforms Linear Regression by **9.4% in R2 score**
- Feature engineering from Project 1 directly contributed to model accuracy
- `total_area` (engineered feature) ranked **#2 in feature importance**
- Ensemble models (Random Forest, XGBoost) significantly outperform
  linear models on tabular data with non-linear relationships
- Model selection should always be based on multiple metrics
  not just one number

---

## Connection to Project 1

This project is built directly on top of:
**[house-price-ds](https://github.com/mehakchanna04-coder/house-price-ds)**
— Advanced EDA & Feature Engineering

The 5 engineered features from Project 1 were used as input features here,
with `total_area` proving to be the 2nd most predictive feature overall.

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Linear Regression, Random Forest, train/test split |
| XGBoost | Gradient boosting model |
| Matplotlib / Seaborn | Visualization |
| Pickle | Model serialization |
| Google Colab | Development environment |
| Git + GitHub | Version control |

---

## Author

**Mehak Channa**
DecodeLabs Industrial Training — Batch 2026
GitHub: [@mehakchanna04-coder](https://github.com/mehakchanna04-coder)

**Project 1:** [house-price-ds](https://github.com/mehakchanna04-coder/house-price-ds)
**Project 2:** [house-price-model](https://github.com/mehakchanna04-coder/house-price-model)

---

*"Mastery of machine learning engineering is not just model tuning.
It is the deployment of strict statistical controls, highly optimized
RAM-based vectorization, and automated contract validation."*
*— DecodeLabs Project 2*
