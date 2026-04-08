from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
model   = pickle.load(open("model.pkl",   "rb"))
scaler  = pickle.load(open("scaler.pkl",  "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ─── Feature importance (coefficients) ───────────────────────────────────────
coef_data = []
for col, coef in zip(columns, model.coef_[0]):
    coef_data.append({"feature": col, "importance": round(abs(coef), 4)})
coef_data.sort(key=lambda x: x["importance"], reverse=True)
TOP_FEATURES = coef_data[:10]

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for col in columns:
        value = request.form.get(col, "0")
        try:
            input_data.append(float(value) if value != "" else 0.0)
        except (ValueError, TypeError):
            input_data.append(0.0)

    raw_input    = np.array([input_data])
    scaled_input = scaler.transform(raw_input)

    prediction  = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    prob_default = round(float(probability[1]) * 100, 2)
    prob_safe    = round(float(probability[0]) * 100, 2)
    result_label = "High Risk" if prediction == 1 else "Low Risk"
    risk_level   = "high" if prediction == 1 else "low"

    return render_template(
        "result.html",
        result=result_label,
        risk_level=risk_level,
        prob_default=prob_default,
        prob_safe=prob_safe,
        top_features=TOP_FEATURES[:5],
    )

@app.route("/api/stats")
def api_stats():
    """Returns model feature importance JSON for dashboard."""
    return jsonify({
        "features": [f["feature"] for f in TOP_FEATURES],
        "importances": [f["importance"] for f in TOP_FEATURES],
    })

if __name__ == "__main__":
    app.run(debug=True)
