import joblib

model = joblib.load("models/mlp_model.pkl")

print(type(model))
print(model)