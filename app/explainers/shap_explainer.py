import shap
import pickle
import pandas as pd


with open("app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.Explainer(model)

feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

def explain_shap(input_df):
    shap_values = explainer.shap_values(input_df)

    if hasattr(shap_values, "values"):
        values = shap_values.values[0]
    else:
        values = shap_values[0]

    
    explanation = {
        feature: float(value)
        for feature, value in zip(feature_names, values)
    }

    explanation = dict(
        sorted(
            explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    )
    return explanation


def generate_reasoning(explaination_dict):
    feature_meaning = {
        "MedInc": "median income",
        "HouseAge": "house age",
        "AveRooms": "average number of rooms",
        "AveBedrms": "average bedrooms",
        "Population": "population in the area",
        "AveOccup": "average occupancy",
        "Latitude": "location (latitude)",
        "Longitude": "location (longitude)"
    }

    reasons = []

    for feature, value in list(explaination_dict.items())[:3]:
        name = feature_meaning.get(feature, feature)

        if value > 0:
            if abs(value) > 1:
                reasons.append(f"High {name} strongly increased the house price")
            else:
                reasons.append(f"{name} slightly increased the house price")
        else:
            if abs(value) > 1:
                reasons.append(f"{name} strongly decreased the house price")
            else:
                reasons.append(f"{name} slightly decreased the house price")

    return reasons