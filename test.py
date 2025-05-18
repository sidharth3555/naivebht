import pickle

with open("best_kidney_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model type:", type(model))
print("Expected input shape:", model.n_features_in_ if hasattr(model, "n_features_in_") else "Unknown")
