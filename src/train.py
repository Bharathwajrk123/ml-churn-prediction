import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib

# Configuration
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/model.pkl"

def download_data():
    """Downloads the Telco Customer Churn dataset if it doesn't exist."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")
    else:
        print("Data already exists.")

def load_and_preprocess_data():
    """Loads and preprocesses the dataset - using only numeric features."""
    df = pd.read_csv(DATA_PATH)
    
    # Select only the 3 numeric features we need for prediction
    df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    
    # Convert TotalCharges to numeric, coercing errors to NaN, then drop NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Convert target variable to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Standardize column names for consistency
    df.columns = ['tenure', 'monthly_charges', 'total_charges', 'Churn']
    
    X = df[['tenure', 'monthly_charges', 'total_charges']]
    y = df['Churn']
    
    # Keep track of feature names for the API
    feature_names = X.columns.tolist()
    
    print(f"Features: {feature_names}")
    print(f"Total samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, scaler):
    """Trains the model and logs metrics/parameters to MLflow."""
    # Start MLflow run
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Customer_Churn_Prediction")

    with mlflow.start_run():
        print("Started MLflow run...")
        
        # Hyperparameters
        n_estimators = 100
        max_depth = 5
        random_state = 42
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name="churn-prediction-model")
        
        # Save artifacts locally
        os.makedirs("models", exist_ok=True)
        artifacts = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names
        }
        joblib.dump(artifacts, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Log additional metadata
        mlflow.log_text(f"Features used: {', '.join(feature_names)}", "features.txt")
        
        return model, scaler

def main():
    """Main pipeline orchestrator."""
    print("=" * 50)
    print("Customer Churn Prediction ML Pipeline")
    print("=" * 50)
    
    # Step 1: Download data
    print("\n[Step 1] Downloading data...")
    download_data()
    
    # Step 2: Load and preprocess
    print("\n[Step 2] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Step 3: Train and evaluate
    print("\n[Step 3] Training model with MLflow tracking...")
    model, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, scaler)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
