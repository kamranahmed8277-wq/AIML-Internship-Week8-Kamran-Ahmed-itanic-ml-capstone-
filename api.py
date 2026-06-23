# app/api.py
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd
import os

# --- Input validation model ---
class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Sex: 'male' or 'female'")
    Age: float = Field(..., ge=0, le=100, description="Age in years (0-100)")
    SibSp: int = Field(0, ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(0, ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Ticket fare (≥0)")
    Embarked: Optional[str] = Field("S", description="Port of embarkation: C, Q, or S")
    Title: Optional[str] = Field("Mr", description="Title (Mr, Mrs, Miss, Master, Rare)")
    FamilySize: Optional[int] = None
    IsAlone: Optional[int] = None
    AgeGroup: Optional[str] = None
    Deck: Optional[str] = None

    @validator('Sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Sex must be "male" or "female"')
        return v.lower()

    @validator('Embarked')
    def validate_embarked(cls, v):
        if v is None:
            return "S"
        if v.upper() not in ['C', 'Q', 'S']:
            raise ValueError('Embarked must be C, Q, or S')
        return v.upper()

# --- Load model from pickle at startup ---
MODEL_PATH = "models/titanic_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run save_champion_model.py first.")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Titanic Survival Prediction API", version="1.0")

# --- Health check ---
@app.get("/health")
async def health():
    return {"status": "healthy", "model_name": "titanic_survival", "version": "1.0"}

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(passenger: PassengerInput):
    # Build the feature DataFrame in the exact order the pipeline expects
    # Order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, IsAlone, AgeGroup, Deck
    data = {
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked,
        "Title": passenger.Title if passenger.Title else "Mr",
        "FamilySize": passenger.FamilySize if passenger.FamilySize is not None else passenger.SibSp + passenger.Parch + 1,
        "IsAlone": passenger.IsAlone if passenger.IsAlone is not None else int(passenger.SibSp + passenger.Parch == 0),
        "AgeGroup": passenger.AgeGroup if passenger.AgeGroup else "Adult",
        "Deck": passenger.Deck if passenger.Deck else "U"
    }
    feature_order = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "AgeGroup", "Deck"]
    df = pd.DataFrame([data])[feature_order]

    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "survived": pred,
        "probability": round(float(proba), 4),
        "verdict": "Survived" if pred == 1 else "Did not survive"
    }