from fastapi import FastAPI
from pydantic_models import CustomerData, PredictionResponse
import mlflow.sklearn
import numpy as np

app = FastAPI()

# Load the model from MLflow
model_uri = "models:/best_rf_model/Production"  # this assumes you registered it
model = mlflow.sklearn.load_model(model_uri)

@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerData):
    features_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict_proba(features_array)[0][1]  # class 1 probability
    return PredictionResponse(risk_probability=round(float(prediction), 4))
