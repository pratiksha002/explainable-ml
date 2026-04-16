from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import pickle

with open("app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

x_train = pd.read_csv("data/x_train.csv")

explainer = LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns.tolist(),
    mode="regression"
)

def explain_lime(input_df):
    instance = input_df.iloc[0].values

    exp = explainer.explain_instance(
        instance,
        model.predict
    )

    return dict(exp.as_list())