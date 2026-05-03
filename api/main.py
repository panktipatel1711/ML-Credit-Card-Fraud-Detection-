from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Real-Time Fraud Detection API")

# Load trained assets
model = joblib.load('models/fraud_model.joblib')
scaler = joblib.load('models/scaler.joblib')

class Transaction(BaseModel):
    features: list # Expecting 29 features (V1-V28 + Amount)

@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array(data.features).reshape(1, -1)
    
    # Get Probability and Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "prediction": "FRAUD" if prediction == 1 else "NORMAL",
        "fraud_probability": f"{round(probability * 100, 2)}%"
    }