from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def home():
    """Root endpoint - serves the churn prediction UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer Churn Prediction</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 50px;
                max-width: 500px;
                width: 100%;
            }

            .header {
                text-align: center;
                margin-bottom: 40px;
            }

            .header h1 {
                color: #333;
                font-size: 28px;
                margin-bottom: 10px;
            }

            .header p {
                color: #666;
                font-size: 14px;
            }

            .form-group {
                margin-bottom: 25px;
            }

            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 14px;
            }

            input {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                transition: all 0.3s ease;
            }

            input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }

            input::placeholder {
                color: #999;
            }

            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 10px;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }

            button:active {
                transform: translateY(0);
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .loader {
                display: none;
                text-align: center;
                margin: 20px 0;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .result {
                display: none;
                margin-top: 30px;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .result.churn {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }

            .result.no-churn {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }

            .result h2 {
                font-size: 24px;
                margin-bottom: 10px;
            }

            .result p {
                font-size: 14px;
                opacity: 0.9;
            }

            .error {
                display: none;
                margin-top: 20px;
                padding: 15px;
                background-color: #fee;
                border-left: 4px solid #f44;
                border-radius: 4px;
                color: #c33;
                font-size: 14px;
                animation: slideIn 0.3s ease;
            }

            .input-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }

            @media (max-width: 480px) {
                .container {
                    padding: 30px 20px;
                }

                .header h1 {
                    font-size: 24px;
                }

                .input-group {
                    grid-template-columns: 1fr;
                }
            }

            .info-box {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                margin-top: 25px;
                font-size: 13px;
                color: #666;
                line-height: 1.6;
            }

            .info-box strong {
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Churn Prediction</h1>
                <p>Predict customer likelihood to churn</p>
            </div>

            <form id="predictionForm">
                <div class="input-group">
                    <div class="form-group">
                        <label for="tenure">Tenure (months)</label>
                        <input 
                            type="number" 
                            id="tenure" 
                            placeholder="e.g., 12" 
                            min="0" 
                            step="1"
                            required
                        >
                    </div>
                    <div class="form-group">
                        <label for="monthlyCharges">Monthly Charges ($)</label>
                        <input 
                            type="number" 
                            id="monthlyCharges" 
                            placeholder="e.g., 70" 
                            min="0" 
                            step="0.01"
                            required
                        >
                    </div>
                </div>

                <div class="form-group">
                    <label for="totalCharges">Total Charges ($)</label>
                    <input 
                        type="number" 
                        id="totalCharges" 
                        placeholder="e.g., 840" 
                        min="0" 
                        step="0.01"
                        required
                    >
                </div>

                <button type="submit" id="predictBtn">Predict Churn</button>
            </form>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #667eea; font-size: 14px;">Analyzing...</p>
            </div>

            <div class="result" id="result">
                <h2 id="resultText"></h2>
                <p id="resultDetails"></p>
            </div>

            <div class="error" id="error">
                <strong>Error:</strong> <span id="errorMessage"></span>
            </div>

            <div class="info-box">
                <strong>ℹ️ How to use:</strong><br>
                Enter customer tenure (in months), monthly charges, and total charges. The model will predict whether the customer is likely to churn (cancel service).
            </div>
        </div>

        <script>
            const form = document.getElementById('predictionForm');
            const loader = document.getElementById('loader');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            const resultText = document.getElementById('resultText');
            const resultDetails = document.getElementById('resultDetails');
            const errorMessage = document.getElementById('errorMessage');
            const predictBtn = document.getElementById('predictBtn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const tenure = parseFloat(document.getElementById('tenure').value);
                const monthlyCharges = parseFloat(document.getElementById('monthlyCharges').value);
                const totalCharges = parseFloat(document.getElementById('totalCharges').value);

                error.style.display = 'none';
                result.style.display = 'none';
                loader.style.display = 'block';
                predictBtn.disabled = true;

                try {
                    const payload = {
                        tenure,
                        monthly_charges: monthlyCharges,
                        total_charges: totalCharges
                    };

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) {
                        throw new Error(`API Error: ${response.status} ${response.statusText}`);
                    }

                    const data = await response.json();
                    loader.style.display = 'none';

                    const prediction = data.prediction;
                    const isChurn = prediction === 1 || prediction === 'Churn';
                    
                    resultText.textContent = isChurn ? '⚠️ Customer Will Churn' : '✅ Customer Will Stay';
                    resultDetails.textContent = isChurn 
                        ? 'High likelihood of service cancellation' 
                        : 'Low likelihood of service cancellation';

                    result.className = isChurn ? 'result churn' : 'result no-churn';
                    result.style.display = 'block';

                } catch (err) {
                    loader.style.display = 'none';
                    errorMessage.textContent = err.message || 'Failed to connect to the API.';
                    error.style.display = 'block';
                    console.error('Error:', err);
                } finally {
                    predictBtn.disabled = false;
                }
            });

            const inputs = document.querySelectorAll('input');
            inputs.forEach(input => {
                input.addEventListener('input', () => {
                    if (error.style.display !== 'none') {
                        error.style.display = 'none';
                    }
                });
            });
        </script>
    </body>
    </html>
    """

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
