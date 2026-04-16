from fastapi import FastAPI
import pickle
import pandas as pd
from app.schemes import HouseData
from app.explainers.shap_explainer import explain_shap
from app.explainers.shap_explainer import generate_reasoning

app = FastAPI()

with open("app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Explainable ML API is running"}

@app.post("/explain")
def explain(data: HouseData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]
        shap_values = explain_shap(df)

        reasons = generate_reasoning(shap_values)

        return {
            "prediction": float(prediction),
            "shap_values": shap_values,
            "top_reasons": reasons
        }
    
    except Exception as e:
        return{
            "error": str(e)
        }