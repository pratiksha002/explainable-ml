import shap
import pickle
import pandas as pd

with open("app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer

def explain_shap(input_df):
    shap_values = explainer.shap_values(input_df)

    return shap_values[0].tolist()