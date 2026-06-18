# 🏦 FinTech Customer Churn AI (End-to-End MLOps Pipeline)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-purple.svg)

## 📌 Executive Summary
Customer churn is one of the most expensive problems in the banking and financial technology (FinTech) industry. Acquiring a new customer costs significantly more than retaining an existing one. 

This project is a fully automated, end-to-end **Machine Learning Web Application** designed to predict bank customer churn. It takes raw customer demographics and financial behavior, processes the data through a custom MLOps pipeline, and utilizes an optimized **XGBoost** algorithm to flag high-risk customers before they leave the institution.

---

## 🌐 Live Web Application
The machine learning model has been fully deployed to the cloud using Render and a custom Flask web server.

**[🚀 Try the Live FinTech AI Here](https://mlproject-01.onrender.com/)**

*(Note: This application is hosted on a free Render tier. If the server has been inactive, it may take up to 60 seconds to "wake up" upon your first visit. Once awake, predictions are generated instantly.)*

---

## 🚀 The Business Value: Precision vs. Recall
In financial modeling, pure accuracy is a vanity metric. Because roughly 80% of customers stay and only 20% leave, a baseline model that guesses "Stay" every single time is mathematically 80% accurate, but financially useless.

This model was specifically engineered to optimize **Recall** (catching actual churners) over pure Precision, directly addressing the high financial cost of a False Negative (a customer closing their account undetected).

**Key Engineering Decisions:**
1. **Target Balancing (SMOTE):** Addressed the severe 80/20 class imbalance by generating synthetic minority data points during training. This prevented the algorithm from developing a majority-class bias.
2. **Threshold Moving (The Secret Weapon):** Shifted the classification probability threshold from the default `0.50` down to `0.40`. 
   * *Result:* Increased the model's ability to successfully catch fleeing customers to **63%** (up from the baseline 56%), prioritizing early retention intervention over minor false-alarm costs.
3. **Hyperparameter Tuning:** Utilized `GridSearchCV` to map the optimal tree depth, learning rate, and estimator count for the XGBoost mathematical engine.

---

## 🏗️ MLOps Architecture
This project graduates from experimental Jupyter Notebooks to a modular, production-ready pipeline architecture suitable for enterprise deployment.

* **`data_ingestion.py`:** Automates the extraction of raw database records and performs a Stratified Train-Test split to ensure distribution parity.
* **`data_transformation.py`:** A highly complex module that handles Categorical Encoding (`OneHotEncoder`), Feature Scaling (`StandardScaler`), and SMOTE balancing, exporting the mathematical translators as `.pkl` artifacts.
* **`model_trainer.py`:** Ingests the transformed matrices, trains the tuned XGBoost algorithm, evaluates the custom 0.40 probability threshold, and serializes the final intelligent model.
* **`predict_pipeline.py`:** The "Universal Translator." It intercepts raw web text from the Flask UI, perfectly mimics the mathematical transformations of the training phase, and feeds the engine to generate live predictions.

---

## 💻 Tech Stack
* **Core Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Backend Web Server:** Flask, Werkzeug
* **Frontend UI:** HTML5, Bootstrap 5 CSS

---

## 📂 Repository Structure
```text
📦 mlproject_01
 ┣ 📂 artifacts               # Serialized models and processed data (.pkl, .csv)
 ┣ 📂 data                    # Raw CSV data
 ┣ 📂 notebooks               # Jupyter notebooks for EDA and initial model selection
 ┣ 📂 src                     # Production Source Code
 ┃ ┣ 📂 components            # MLOps factory machines (Ingestion, Transformation, Training)
 ┃ ┣ 📂 pipeline              # Automation scripts (train_pipeline, predict_pipeline)
 ┃ ┣ 📜 utils.py              # Universal toolbox (saving/loading objects)
 ┣ 📂 templates               # HTML Frontend (index.html)
 ┣ 📜 app.py                  # Flask Web Server
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 README.md               # Project documentation

```

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
```bash
git clone [https://github.com/Tapiwanasche/mlproject_01.git](https://github.com/Tapiwanasche/mlproject_01.git)
cd mlproject_01

```


2. **Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Mac/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

```


3. **(Optional) Run the automated training pipeline to build a fresh model:**
```bash
python -m src.pipeline.train_pipeline

```


4. **Start the Flask Web Server:**
```bash
python app.py

```


5. **Open your browser:**
Navigate to `http://127.0.0.1:5000` to interact with the AI directly.

---

*Developed by Tapiwanashe J Tapfumaneyi.*

```

```