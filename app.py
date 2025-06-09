from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import os

app = Flask(__name__)

# Fungsi untuk mengunduh file dari URL
def download_file(url, dest):
    if not os.path.exists(dest):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()  # Cek jika ada error HTTP
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Berhasil mengunduh {dest}")
        except Exception as e:
            raise Exception(f"Gagal mengunduh {dest}: {e}")

# Unduh model dan scaler dari Google Drive
model_path = "klasifikasidisorder.pkl"
scaler_path = "scalerklasifikasidisorder.pkl"
download_file("https://drive.google.com/uc?export=download&id=1L6Noe6IyqKzLkIJqXZxRYSbR9IFxboJC", model_path)
download_file("https://drive.google.com/uc?export=download&id=1UVUoPv86OlWw4Bu2xHAkYSCVHOv6yX-1", scaler_path)

# Load model dan scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise Exception(f"Error loading model or scaler: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sleep Disorder Prediction API"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required_fields = [
            "Gender_cod", "Age", "Occupation_cod", "Sleep Duration",
            "Quality of Sleep", "Physical Activity Level", "Stress Level",
            "BMI Category_cod", "Heart Rate", "Daily Steps"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

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
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)
        pred_label = "Disorder" if prediction[0] == 1 else "No Disorder"
        return jsonify({"prediction": pred_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)