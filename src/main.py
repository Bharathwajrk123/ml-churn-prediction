from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API that predicts customer churn based on Telco data.",
    version="1.0.0"
)

# Global variables to hold the model and scaler
model = None
scaler = None
feature_names = None

MODEL_PATH = "models/model.pkl"

@app.on_event("startup")
def load_artifacts():
    """Loads the trained model and scaler on API startup."""
    global model, scaler, feature_names
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_names = artifacts["feature_names"]
        print("Model and scaler loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoints will fail.")

# We dynamically create a Pydantic model for input based on the expected features
class ChurnPredictionRequest(BaseModel):
    # These are some common features from the Telco dataset after preprocessing
    # Since we one-hot encoded in train.py, the exact feature names depend on the dataset.
    # For a robust production API, we'd typically map these carefully or use a dict.
    # To handle dynamic inputs seamlessly, we'll accept a dictionary of features.
    features: dict

class ChurnPredictionResponse(BaseModel):
    prediction: int
    probability_churn: float
    probability_no_churn: float

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    """Predicts customer churn given a set of features."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert input dictionary to DataFrame
        # It needs to have exactly the same columns as the training data
        input_data = pd.DataFrame([request.features])
        
        # Ensure all columns are present, filled with 0 if missing (for one-hot encoded columns)
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Reorder columns to match training exactly
        input_data = input_data[feature_names]
        
        # Scale the numerical features (scaler expects the same columns in the same order)
        scaled_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        
        return ChurnPredictionResponse(
            prediction=int(prediction),
            probability_no_churn=float(probabilities[0]),
            probability_churn=float(probabilities[1])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    """Root endpoint - API information."""
    return {
        "message": "ML Churn Prediction API is running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "POST /predict": "Make a churn prediction",
            "GET /health": "Health check",
            "GET /version": "API and model version info"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/version")
def version_info():
    """API version and model information."""
    return {
        "model_version": "1.0",
        "api_version": "1.0.0",
        "model_name": "Random Forest Classifier",
        "dataset": "Telco Customer Churn"
    }
