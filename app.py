from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "🚀 ML Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})
