from flask import Flask, request, jsonify # pyright: ignore[reportMissingImports]
import joblib # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "model.pkl")

model = joblib.load(model_path)

@app.route("/")
def home():
    return "Student Performance Prediction API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[ 
        data["study_hours"], 
        data["attendance"], 
        data["previous_score"]
    ]])
    prediction = model.predict(features)
    return jsonify({"Predicted Score": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
