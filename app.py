from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model dan scaler
try:
    model = joblib.load("klasifikasidisorder.pkl")
    scaler = joblib.load("scalerklasifikasidisorder.pkl")
except Exception as e:
    raise Exception(f"Error loading model or scaler: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sleep Disorder Prediction API"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.get_json()
        
        # Validasi input
        required_fields = [
            "Gender_cod", "Age", "Occupation_cod", "Sleep Duration",
            "Quality of Sleep", "Physical Activity Level", "Stress Level",
            "BMI Category_cod", "Heart Rate", "Daily Steps"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Buat DataFrame dari input
        input_data = {
            "Gender_cod": data["Gender_cod"],
            "Age": data["Age"],
            "Occupation_cod": data["Occupation_cod"],
            "Sleep Duration": data["Sleep Duration"],
            "Quality of Sleep": data["Quality of Sleep"],
            "Physical Activity Level": data["Physical Activity Level"],
            "Stress Level": data["Stress Level"],
            "BMI Category_cod": data["BMI Category_cod"],
            "Heart Rate": data["Heart Rate"],
            "Daily Steps": data["Daily Steps"]
        }
        df = pd.DataFrame([input_data])

        # Standarisasi data
        scaled_data = scaler.transform(df)

        # Prediksi
        prediction = model.predict(scaled_data)
        pred_label = "Disorder" if prediction[0] == 1 else "No Disorder"

        return jsonify({"prediction": pred_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)