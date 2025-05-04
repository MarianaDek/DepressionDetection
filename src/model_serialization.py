import joblib

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    return joblib.load(path)
