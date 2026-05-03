# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
This project implements a robust **AI-Powered Fraud Detection System** designed to identify fraudulent credit card transactions. Using a Random Forest Classifier, the system achieves high precision in detecting anomalies within highly imbalanced financial datasets.

## 🏗️ System Architecture
The system follows a standard data science pipeline from ingestion to deployment:
![System Architecture](images/architecture_diag.png)

## 📊 Model Performance
To ensure reliability, the model was evaluated using a Confusion Matrix, focusing on minimizing False Negatives (missed fraud cases).
![Confusion Matrix](images/confusion_matrix.png)

### Key Metrics:
- **Algorithm:** Random Forest Classifier
- **Features:** PCA-transformed transaction features
- **Handling Imbalance:** Tested with SMOTE/Standard Scaling

## 📂 Project Structure
```text
├── data/               # Raw and processed datasets
├── images/             # System architecture and performance plots
├── models/             # Serialized .joblib model and scaler files
├── notebooks/          # Modularized research and training steps
├── app.py              # Streamlit Dashboard interface
└── requirements.txt    # Project dependencies
