import pandas as pd
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing(as_frame=True)
df = data.frame

x = df.drop("MedHouseVal", axis = 1)
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

with open("app/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

x_train.to_csv("data/x_train.csv", index=False)

print("Model trained and saved")