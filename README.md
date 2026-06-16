# Bank Customer Churn Prediction: End-to-End ML Pipeline

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict bank customer churn. By identifying customers at high risk of leaving the bank, financial institutions can proactively deploy targeted retention strategies, thereby reducing customer acquisition costs and stabilizing revenue. 

This repository demonstrates a complete MLOps-ready architecture, moving from exploratory data analysis (EDA) to modular feature engineering, model training, and evaluation.

## 💼 The Business Problem
Customer retention is significantly more cost-effective than customer acquisition. The objective of this model is to analyze demographic and financial behaviors to flag at-risk customers before they close their accounts. 

**Key EDA Insights Discovered:**
* **Class Imbalance:** The dataset exhibits a ~20% churn rate, establishing a baseline where standard accuracy metrics are insufficient.
* **Geographic Risk:** Customers in Germany churn at a disproportionately higher rate compared to France and Spain.
* **Feature Importance:** Age (positive correlation) and Active Member status (negative correlation) are the strongest linear predictors of customer exit.

## 🏗️ Project Architecture
The codebase is structured following professional software engineering and MLOps principles, ensuring modularity, reproducibility, and scalability.

```text
MLPROJECT_01/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and experimental preprocessing
│   ├── 1_EDA_and_Data_Preparation.ipynb
│   └── 2_Feature_Engineering.ipynb
├── src/                    # Modular source code for pipeline execution
│   ├── components/         # Data ingestion, transformation, and model training scripts
│   ├── pipeline/           # Training and prediction pipelines
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Execution logging
│   └── utils.py            # Shared utility functions
├── requirements.txt        # Project dependencies
└── setup.py                # Package initialization