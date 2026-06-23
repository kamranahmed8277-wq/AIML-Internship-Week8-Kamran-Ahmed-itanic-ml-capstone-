import pytest
from fastapi.testclient import TestClient
from api import app          # <--- import from api.py

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "titanic_survival"

def test_valid_young_female_first_class():
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "C"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["survived"] == 1
    assert data["probability"] > 0.5

def test_valid_older_male_third_class():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 50,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 10.0,
        "Embarked": "S"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["survived"] == 0
    assert data["probability"] < 0.5

def test_invalid_pclass():
    payload = {"Pclass": 5, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 20.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_missing_required_field():
    payload = {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0, "Fare": 100.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422