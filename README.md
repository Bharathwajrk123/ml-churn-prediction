# 🤖 Customer Churn Prediction - Production ML Project

A **production-ready Machine Learning project** that predicts customer churn using scikit-learn, MLflow, FastAPI, and Docker. This project demonstrates best practices for building, tracking, containerizing, and deploying ML models to the cloud.

---

## 📋 Project Overview

### Problem Statement
Customer churn is a critical business metric. This project builds a predictive model to identify customers likely to leave, enabling proactive retention strategies.

### Dataset
- **Source**: Telco Customer Churn dataset
- **Features**: 20+ customer features (tenure, contracts, services, charges, etc.)
- **Target**: Binary classification (Churn: Yes/No)
- **Size**: ~7,000 customer records

### Technology Stack
- **ML Framework**: scikit-learn (Random Forest)
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker & Docker Compose
- **Deployment**: Railway
- **Language**: Python 3.10+

---

## 🏗️ Project Structure

```
ml_project/
├── data/                    # Datasets
│   └── churn.csv           # Telco Customer Churn data (auto-downloaded)
├── src/                    # Source code
│   ├── main.py            # FastAPI REST API application
│   └── train.py           # Training pipeline with MLflow
├── models/                # Saved model artifacts
│   └── model.pkl          # Serialized model, scaler, and features
├── notebooks/             # Jupyter notebooks for exploration
│   └── exploration.ipynb  # EDA and analysis
├── mlruns/                # MLflow tracking database
│   └── mlflow.db          # SQLite database
├── tests/                 # Unit tests
│   └── test_api.py        # API endpoint tests
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Multi-container orchestration
├── .gitignore             # Git ignore rules
└── README.md             # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- Docker & Docker Compose (optional, for containerization)
- Git (for Railway deployment)

### 1️⃣ Local Installation

#### Option A: Virtual Environment

```bash
# Clone repository
git clone <your-repo-url>
cd ml_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Conda

```bash
conda create -n churn-prediction python=3.10
conda activate churn-prediction
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
# Run the training pipeline
python src/train.py

# This will:
# 1. Download the Telco Customer Churn dataset
# 2. Preprocess and scale features
# 3. Train a Random Forest model
# 4. Log metrics and parameters to MLflow
# 5. Save the model to models/model.pkl
```

**Expected Output:**
```
==================================================
Customer Churn Prediction ML Pipeline
==================================================

[Step 1] Downloading data...
Data already exists.

[Step 2] Loading and preprocessing data...
Training set shape: (5636, 44)
Test set shape: (1409, 44)
Number of features: 44

[Step 3] Training model with MLflow tracking...
Started MLflow run...
Accuracy: 0.8234
Precision: 0.6789
Recall: 0.7234

==================================================
Pipeline completed successfully!
==================================================
```

### 3️⃣ View MLflow Dashboard

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Open browser: http://localhost:5000
```

In the MLflow dashboard, you can:
- View experiment runs and metrics
- Compare model performance across runs
- Track parameters and hyperparameters
- Access logged models

### 4️⃣ Start the API Locally

```bash
# Run the FastAPI server
uvicorn src.main:app --reload

# API available at: http://localhost:8000
# Interactive API docs: http://localhost:8000/docs
# Alternative docs: http://localhost:8000/redoc
```

---

## 📡 API Usage

### Endpoints

#### 1. **Health Check** (`GET /health`)
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. **Predict Churn** (`POST /predict`)

Send a JSON request with customer features:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure_months": 24,
      "monthly_charges": 65.5,
      "total_charges": 1570.0,
      "contract_one_year": 1,
      "online_security_yes": 1
    }
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability_churn": 0.78,
  "probability_no_churn": 0.22
}
```

**Interpretation:**
- `prediction`: 1 = Customer likely to churn, 0 = Customer likely to stay
- `probability_churn`: Confidence of churn (0-1)
- `probability_no_churn`: Confidence of staying (0-1)

#### 3. **API Documentation** 

Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

## � Docker Setup

### Build and Run Locally

#### Single Container (API Only)

```bash
# Build Docker image
docker build -t churn-prediction:latest .

# Run container
docker run -p 8000:8000 churn-prediction:latest

# API available at: http://localhost:8000
```

#### Full Stack with Docker Compose

```bash
# Start all services (API + MLflow)
docker-compose up --build

# Access:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000
```

**Stop Services**
```bash
docker-compose down

# Remove volumes (clears MLflow data)
docker-compose down -v
```

### Docker Compose Structure

The `docker-compose.yml` includes:

1. **API Service** (`api`)
   - Runs FastAPI uvicorn server
   - Port: 8000
   - Volumes: src/ and models/ (for development)
   - Depends on MLflow service

2. **MLflow Service** (`mlflow`)
   - Runs MLflow tracking server
   - Port: 5000
   - Backend: SQLite (mlruns/mlflow.db)
   - Artifact root: /mlruns

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Manual API Testing

#### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict churn (example with common features)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure_months": 6,
      "monthly_charges": 85.0,
      "total_charges": 510.0,
      "contract_month_to_month": 1,
      "internet_service_fiber_optic": 1,
      "tech_support_no": 1
    }
  }'
```

#### Using Postman

1. Create a new POST request
2. URL: `http://localhost:8000/predict`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "features": {
    "tenure_months": 24,
    "monthly_charges": 65.5,
    "total_charges": 1570.0,
    "contract_one_year": 1,
    "online_security_yes": 1
  }
}
```

#### Using Python Requests

```python
import requests
import json

url = "http://localhost:8000/predict"
payload = {
    "features": {
        "tenure_months": 36,
        "monthly_charges": 75.0,
        "total_charges": 2700.0
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 🚢 Railway Deployment

### Step 1: Prepare GitHub Repository

```bash
# Initialize git (if not already done)
git init

# Create .gitignore
cat > .gitignore << EOF
# Virtual environment
venv/
env/
ENV/

# MLflow and model artifacts
mlruns/
models/
*.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Local development
.env
.env.local
EOF

# Add and commit all files
git add .
git commit -m "Initial commit: Customer Churn Prediction ML project"

# Add remote and push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/ml_project.git
git branch -M main
git push -u origin main
```

### Step 2: Create Railway Project

1. **Sign up on Railway**
   - Go to: https://railway.app
   - Sign up with GitHub (recommended)

2. **Create a New Project**
   - Click "New Project" → "Deploy from GitHub"
   - Authorize Railway to access GitHub
   - Select your `ml_project` repository
   - Click "Deploy"

### Step 3: Configure Environment Variables

In Railway dashboard:

1. Go to your project → Variables
2. Add the following variables:
   ```
   PORT=8000
   MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
   PYTHONUNBUFFERED=1
   ```

3. **Optional: Configure for Production**
   ```
   DEBUG=False
   ENV=production
   ```

### Step 4: Deploy

Railway automatically deploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update: improve model performance"

# Push to GitHub
git push origin main

# Railway automatically builds and deploys
# Check deployment status at: https://railway.app/dashboard
```

### Step 5: Test Deployed API

```bash
# Get your Railway URL from dashboard (e.g., https://churn-prediction-prod.railway.app)
RAILWAY_URL="https://your-project.railway.app"

# Health check
curl $RAILWAY_URL/health

# Test prediction
curl -X POST "$RAILWAY_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure_months": 12,
      "monthly_charges": 55.0,
      "total_charges": 660.0
    }
  }'
```

### Step 6: View Logs (Debugging)

In Railway dashboard:
- Go to your project → Deployments → Select deployment → View Logs
- Monitor application health and errors in real-time

---

## 📊 MLflow Tracking Deep Dive

### What Gets Logged?

The training pipeline logs:

1. **Parameters** (hyperparameters)
   - `n_estimators`: Number of trees
   - `max_depth`: Max tree depth
   - `random_state`: Seed for reproducibility

2. **Metrics** (performance measures)
   - `accuracy`: Overall prediction accuracy
   - `precision`: True positive rate among predicted positives
   - `recall`: True positive rate among actual positives

3. **Models** (artifacts)
   - Trained scikit-learn model
   - Registered in MLflow Model Registry

4. **Metadata** (additional info)
   - Feature names
   - Dataset information

### Accessing MLflow Programmatically

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# Query experiments
experiments = mlflow.search_experiments()
print(experiments)

# Query runs
runs = mlflow.search_runs(experiment_names=["Customer_Churn_Prediction"])
print(runs)

# Load a registered model
model = mlflow.pyfunc.load_model("models:/churn-prediction-model/latest")
predictions = model.predict(data)
```

---

## 🔧 Customization Guide

### Change the Model Type

In `src/train.py`, replace the Random Forest with another sklearn model:

```python
# Original
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5)

# Example: Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, max_depth=3)

# Example: Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
```

### Adjust Hyperparameters

```python
# In src/train.py, train_and_evaluate() function
n_estimators = 200  # Increase for more trees
max_depth = 10      # Increase for deeper trees
random_state = 42   # Change for different splits
```

### Add New Metrics

```python
from sklearn.metrics import f1_score, roc_auc_score

f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)
```

---

## 📚 Learning Resources

### Best Practices Implemented

1. **MLOps**: Experiment tracking with MLflow
2. **API Design**: RESTful API with FastAPI
3. **Containerization**: Docker for reproducibility
4. **Version Control**: Git for code management
5. **CI/CD**: Automated deployment on Railway
6. **Testing**: Unit tests for API endpoints
7. **Documentation**: Clear README and code comments

### Further Reading

- [MLflow Documentation](https://mlflow.org/docs/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Railway Docs](https://docs.railway.app/)

---

## 🐛 Troubleshooting

### Model Not Loading
**Error**: `Warning: Model not found at models/model.pkl`

**Solution**: Run the training pipeline first
```bash
python src/train.py
```

### Port Already in Use
**Error**: `Address already in use: ('0.0.0.0', 8000)`

**Solution**: Use a different port
```bash
uvicorn src.main:app --port 8001
```

### MLflow Connection Error
**Error**: `Cannot connect to tracking server`

**Solution**: Ensure MLflow is running
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Docker Build Fails
**Error**: `COPY src/ /app/src/ failed`

**Solution**: Build from project root
```bash
cd ml_project
docker build -t churn-prediction:latest .
```

### Railway Deployment Fails
**Error**: `Build failed` in Railway logs

**Solution**: Check logs and ensure:
- All files are committed to Git
- requirements.txt is up to date
- PORT environment variable is set
- No secrets in code (use environment variables)

---

## 📝 License

MIT License - Feel free to use and modify for your projects.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information

---

## 🎯 Next Steps

**To extend this project:**

1. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
2. **Feature Engineering**: Add domain-specific features
3. **Data Validation**: Implement data quality checks
4. **Model Monitoring**: Add drift detection and performance tracking
5. **A/B Testing**: Deploy multiple models and compare
6. **Automated Retraining**: Schedule periodic model retraining
7. **API Authentication**: Add API keys or OAuth
8. **Advanced Metrics**: Add business metrics and ROI calculations

---

**Happy modeling! 🚀**
