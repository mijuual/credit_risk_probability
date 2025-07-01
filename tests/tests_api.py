from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Credit Risk Model API is running."

def test_prediction():
    response = client.post("/predict", json={"features": [0.1]*30})  # Adjust length!
    assert response.status_code == 200
    assert "risk_probability" in response.json()
