import pandas as pd
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score

try:
    from src.preprocessing import clean_data, scale_features
except ImportError:
    from preprocessing import clean_data, scale_features

def train_model(data_path):
    print("--- 🚀 Starting Professional Training Pipeline ---")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Preprocess & Scale
    df_cleaned = clean_data(df)
    X, y, scaler = scale_features(df_cleaned)
    
    # 3. Stratified Train-Test Split (Keeps fraud ratio consistent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Handle Imbalance (SMOTE)
    print("🔄 Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 5. XGBoost Model Training
    print("🧠 Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    model.fit(X_res, y_res)

    # 6. Evaluation using Industry Metrics (PR-AUC)
    y_pred = model.predict(X_test)
    print("\n--- 📊 Performance Report ---")
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"⭐ PR-AUC Score: {pr_auc:.4f}")

    # 7. Save Assets for API Deployment
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("\n✅ Success: Model and Scaler saved in 'models/' folder.")

if __name__ == "__main__":
    train_model('data/raw/creditcard.csv')