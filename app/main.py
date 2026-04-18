from fastapi import FastAPI
import pickle
import pandas as pd
from app.schemes import HouseData
from app.explainers.shap_explainer import explain_shap
from app.explainers.shap_explainer import generate_reasoning
from app.explainers.lime_explainer import explain_lime
from app.explainers.custom_explainer import compare_explanations
from app.database.db import collection
from datetime import datetime 

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
        lime_values = explain_lime(df)
        reasons = generate_reasoning(shap_values)
        comparison = compare_explanations(shap_values, lime_values)

        record = {
            "input": data.dict(),
            "prediction": float(prediction),
            "shap_values": shap_values,
            "lime_values": lime_values,
            "top_reasons": reasons,
            "comparison": comparison,
            "timestamp": datetime.now()
        }
        print("Saving to DB...")
        collection.insert_one(record)
        print("Saved")
        return {
            "prediction": float(prediction),
            "shap_values": shap_values,
            "lime_values": lime_values,
            "top_reasons": reasons,
            "comparison": comparison
        }
    
    except Exception as e:
        print("ERROR:", e) 
    return {
        "error": str(e)
    }

@app.get("/history")
def get_history():
    try:
        data = list(collection.find({}, {"_id":0}))

        return{
            "total_records": len(data),
            "records": data
        }
    
    except Exception as e:
        return{
            "error": str(e)
        }
    

@app.delete("/clear-test")
def clear_test():
    collection.delete_many({"test": "working"})
    return {"message": "Test data removed"}