import streamlit as st
import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb+srv://pratiksha:pratiksha1902@explainable-ml.zlnjqoi.mongodb.net/?retryWrites=true&w=majority")
db = client["explainable_ml"]
collection = db["predictions"]

st.title("Explainable ML Dashboard")

data = list(collection.find({}, {"_id":0}))

if len(data) == 0:
    st.write("No data found")

else:
    df = pd.json_normalize(data)

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Predictions Over Time")
    if "prediction" in df.columns:
        st.line_chart(df["prediction"].dropna())

    else:
        st.write("No prediction data available")

    
    st.subheader("Feature Importance (SHAP)")
    valid_data = [d for d in data if "shap_values" in d]

    if len(valid_data) > 0:
        shap_df = pd.DataFrame([d["shap_values"] for d in valid_data])
        shap_mean = shap_df.abs().mean().sort_values(ascending=False)
        st.bar_chart(shap_mean)

    else:
        st.write("No SHAP data available")

    st.subheader("Median Income vs Prediction")
    filtered_df = df.dropna(subset=["input.MedInc", "prediction"])
    
    if not filtered_df.empty:
        st .scatter_chart(filtered_df[["input.MedInc", "prediction"]])
    else:
        st.write("NOt enough data for scatter plot")