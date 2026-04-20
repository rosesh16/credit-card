# IEEE-CIS Credit Card Fraud Detection Pipeline

![Fraud Detection System](https://img.shields.io/badge/Status-Active-brightgreen) ![Python 3.14](https://img.shields.io/badge/Python-3.14-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a682) ![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-orange)

An end-to-end, production-ready machine learning framework engineered to detect fraudulent credit card transactions. Based on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection), this project implements advanced Kaggle-winning feature engineering, chronologically secure validation, and a real-time FastAPI inference engine wrapped in an interactive visual dashboard.

## 🌟 Key Features

### 1. Robust Machine Learning Pipeline
*   **Time-Series Validation:** Uses a strictly chronological train/test split to perfectly mimic real-world deployment and prevent data-leakage from the future.
*   **Imbalance Handling:** The dataset is heavily skewed (96.5% legitimate). The LightGBM classifier natively scales positive weights directly inside the loss function without expensive Smote/Oversampling.
*   **Target Mean Encoding:** Transforms high-cardinality nominal variables into historically mapped fraud probabilities.

### 2. FastAPI Inference Engine
*   **Live Feature Mapping:** At startup, the API loads historical behavior mappings into a "Feature Store" memory map for sub-millisecond dynamic feature calculation.
*   **LIME Explainability:** Automatically unpacks black-box LightGBM probabilities, breaking down exactly which features (e.g., specific missing addresses, abnormal temporal frequencies) influenced the fraud score.

### 3. Interactive Web Dashboard
*   **Dynamic Thresholding:** A slider that allows risk-managers to adjust the precision/recall threshold in real-time, instantly displaying the impact on False Positives and Financial Loss.
*   **Live Curves:** Renders ROC and Precision-Recall evaluation curves via Chart.js dynamically fetched from the model's test-set API endpoints.

---

## 🛠️ Project Structure

```bash
Credit_card_fraud_detection/
├── Data/                 # Raw datasets (Not tracked in version control)
│   ├── raw/
│   └── processed/        # Compiled joblib pipelines and metrics cache
├── frontend/
│   ├── index.html        # Interactive simulation dashboard
│   ├── index.js          # REST integration & Chart.js logic
│   └── index.css         # Modern, dark-mode glassmorphism styling
├── notebooks/
│   └── EDA.ipynb         # Initial Exploratory Data Analysis & SMOTE experiments
├── src/
│   ├── app.py            # FastAPI REST backend & inference
│   ├── train.py          # LightGBM training, mapping extraction, and evaluation
│   └── config.py         # File paths, ML hyperparameters, and test definitions
├── requirements.txt      # Project dependencies
└── .gitignore
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python configured (developed on `3.14`), then create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Install deps
pip install -r requirements.txt
```

### 2. Providing the Data
Download the [IEEE-CIS Fraud Detection Data](https://www.kaggle.com/c/ieee-fraud-detection/data). Place `train_transaction.csv` and `train_identity.csv` inside the `Data/raw/` directory.

### 3. Train the Model
Run the pipeline to extract features, build chronological validation data, and tune the LightGBM classifier:

```bash
python src/train.py
```
*(This will compile mapping `.joblib` files, a `.json` metrics dump, and the main model into `Data/processed/`)*

### 4. Start the Application
Boot up the FastAPI server, which mounts your engineered feature store into memory:

```bash
python src/app.py
```

Finally, simply double-click `frontend/index.html` in your browser to launch the Interactive Risk-Management Dashboard.