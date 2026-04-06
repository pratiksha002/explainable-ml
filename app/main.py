from fastapi import FastAPI
import pickle
import pandas as pd
from app.schemes import HouseData

app = FastAPI()

with open("app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Explainable ML API is running"}

@app.post("/predict")
def predict(data: HouseData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        return {
            "prediction": float(prediction)
        }
    
    except Exception as e:
        return{
            "error": str(e)
        }