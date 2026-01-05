import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def upload_file(file_bytes, filename):
    files = {"file": (filename, file_bytes, "text/csv")}
    res = requests.post(f"{BASE_URL}/upload", files=files)
    return res.json()

def train_model(filename, target, drop_cols=[], auto_optimize=False):
    payload = {
        "filename": filename,
        "target_column": target,
        "drop_columns": drop_cols,
        "auto_optimize": auto_optimize # <--- SEND FLAG
        }
    res = requests.post(f"{BASE_URL}/train", json=payload)
    return res.json()

def get_prediction(features):
    res = requests.post(f"{BASE_URL}/predict", json={"features": features})
    return res.json()

def get_explanation(features, prediction, probability):
    payload = {
        "features": features, 
        "prediction": prediction, 
        "probability": probability
    }
    res = requests.post(f"{BASE_URL}/explain", json=payload)
    return res.json()

def get_counterfactual(features):
    res = requests.post(f"{BASE_URL}/simulate", json={"features": features})
    return res.json()