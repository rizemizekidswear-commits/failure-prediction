from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("failure_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["air_temp"],
        data["process_temp"],
        data["rot_speed"],
        data["torque"],
        data["tool_wear"]
    ]])
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    status = "خراب" if prob >= 0.3 else "سالم"
    return jsonify({
        "احتمال_خرابی": f"{prob*100:.1f}%",
        "وضعیت": status
    })

if __name__ == "__main__":
    app.run(debug=True)
