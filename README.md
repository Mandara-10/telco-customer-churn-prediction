# Telco Customer Churn Prediction - Production Microservices Architecture

This repository contains a complete, production-ready Telco Customer Churn Prediction application decoupled into two distinct microservices:
1. **Flask REST API Backend (`backend/`)**: Serves model inference, preprocessing, validation rules, feature metadata, and SHAP explanations.
2. **Streamlit Frontend (`frontend/`)**: Pure presentation layer built dynamically via backend REST API calls with no direct model/SHAP dependencies.

---

## 🌟 Key Features

- **Decoupled Architecture**: Clean separation between Flask API backend and Streamlit UI frontend.
- **Optimized DNN Model**: Keras Deep Neural Network trained with Dropout (0.5), Batch Normalization, and Adam optimizer (`lr=3e-4`).
- **Custom Classification Threshold**: Optimal threshold of `0.45` tuned for high recall on churn detection.
- **Local & Global SHAP Explainability**: Detailed feature attributions for individual predictions and global model insights.
- **Batch CSV Processing**: Upload CSV files for batch scoring with inline validation error highlighting and CSV exports.
- **Dynamic Form Generation**: Frontend dynamically constructs input controls from `GET /api/features`.

---

## 📊 Dataset & Model Performance

### Dataset Overview
- **Source**: Telecom Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Total Records**: 7,043 rows (19 feature columns + Churn target)

### Model Test Metrics (Threshold = 0.45)

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7821 |
| **Precision** | 0.5781 |
| **Recall** | 0.6631 |
| **F1-Score** | 0.6177 |
| **ROC-AUC** | 0.8397 |

### Model Benchmark Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8420 |
| Decision Tree | 0.7942 | 0.6296 | 0.5455 | 0.5845 | 0.8284 |
| Random Forest | 0.7892 | 0.6305 | 0.4973 | 0.5561 | 0.8226 |
| **Optimized DNN (O10-BatchNorm)** | **0.7821** | **0.5781** | **0.6631** | **0.6177** | **0.8397** |

---

## 📂 Clean Project Structure

```
telco-customer-churn-prediction/
├── backend/
│   ├── app.py                      # Flask REST API app factory & endpoints
│   ├── config.py                   # System & path configurations, threshold (0.45)
│   ├── requirements.txt            # Backend dependencies
│   ├── models/                     # Trained .keras model & fitted .joblib preprocessor
│   │   ├── churn_dnn_optimized.keras
│   │   └── preprocessor_optimized.joblib
│   ├── utils/
│   │   ├── feature_mapping.py      # Single source of truth for 19 features
│   │   ├── validation.py           # Payload & batch CSV validation logic
│   │   └── prediction.py           # ChurnPredictor singleton class
│   └── explainability/
│       ├── shap_explainer.py       # ChurnExplainer class (Deep/Kernel fallbacks)
│       └── shap_background.joblib  # Auto-generated background dataset artifact
│
├── frontend/
│   ├── streamlit_app.py            # Streamlit UI (Single Predict, Batch CSV, Insights)
│   ├── api_client.py               # REST API HTTP wrapper client
│   └── requirements.txt            # Frontend dependencies
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_dnn_optimized.keras
│   └── preprocessor_optimized.joblib
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Baseline_Models.ipynb
│   ├── 05_DNN_Model_Proper.ipynb
│   └── 06_DNN_Optimization.ipynb
│
├── reports/
│   └── ...
│
└── README.md
```

---

## 🚀 How to Run the Application

### 1. Start the Flask Backend API (Port 5000)

Open a terminal in the project root directory and run:

```bash
# Set PYTHONPATH to project root
$env:PYTHONPATH="."  # Windows PowerShell
# export PYTHONPATH="."  # Linux / macOS

python -m backend.app
```

The Flask server will start listening at `http://localhost:5000` and load model singletons into memory.

### 2. Start the Streamlit UI Frontend (Port 8501)

Open a second terminal in the project root directory and run:

```bash
# Set BACKEND_URL (optional, defaults to http://localhost:5000)
$env:BACKEND_URL="http://localhost:5000"

streamlit run frontend/streamlit_app.py --server.port 8501
```

Access the user interface at `http://localhost:8501`.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Liveness and readiness health check |
| `GET` | `/api/features` | Feature metadata for dynamic UI form generation |
| `POST` | `/api/predict` | Single customer prediction payload -> probability & prediction |
| `POST` | `/api/predict/explain` | Single customer prediction payload -> prediction + SHAP explanation |
| `POST` | `/api/predict/batch` | Multipart CSV upload -> predictions CSV/JSON output |
| `POST` | `/api/explain/batch-row` | Batch customer row -> SHAP explanation for specific row |
| `GET` | `/api/model/insights` | Metrics table, model comparison table, global SHAP importance, and limitations |
