"""
========================================================
  ML MODEL DEMO — BACKEND SERVER
  Grup: [Grup Adınız / Group Name]
  Proje: [Proje Başlığı / Project Title]
========================================================

KURULUM / SETUP:
  pip install flask flask-cors joblib scikit-learn numpy pandas

ÇALIŞTIRMA / RUN:
  python demo_server.py
  → Tarayıcıda / Open in browser: http://localhost:5000

NOTLAR / NOTES:
  - Modelinizi joblib veya pickle ile kaydetmiş olmanız gerekir.
  - Model dosyasının yolunu aşağıda MODEL_PATH'e yazın.
  - Feature isimlerinin model eğitimindeki sırayla eşleştiğinden emin olun.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder=".")
CORS(app)  # allows the HTML file to call this server


# ================================================================
#  STUDENT CONFIG — edit this section before your presentation
# ================================================================

MODEL_PATH = "model.joblib"          # path to your saved model file
                                     # e.g. "models/rf_model.pkl" or "xgb_model.joblib"

SCALER_PATH = None                   # path to your scaler if you used one, else None
                                     # e.g. "models/scaler.joblib"

MODEL_TYPE = "regression"            # "regression" or "classification"

# Feature names EXACTLY as used during model training (same order!)
FEATURE_NAMES = [
    "dev_game_count",
    "solo_dev_proxy",
    "dev_equals_publisher",
    "achievement_count",
    "has_demo",
    "dlc_count",
    "genre_count",
    "platform_count",
    "language_count",
    "price_usd",
    "is_early_access",
    "post_chatgpt",
    "release_year",
    "has_workshop",
]

# Human-readable labels for the UI (same order as FEATURE_NAMES)
FEATURE_LABELS = [
    "Geliştirici oyun sayısı (Steam'deki toplam) [dev_game_count]",
    "Solo geliştirici mi? (1=Evet, 0=Hayır) [solo_dev_proxy]",
    "Geliştirici = Yayıncı mı? (1=Evet, 0=Hayır) [dev_equals_publisher]",
    "Başarım sayısı [achievement_count]",
    "Demo var mı? (1=Evet, 0=Hayır) [has_demo]",
    "DLC sayısı [dlc_count]",
    "Tür sayısı [genre_count]",
    "Platform sayısı (Windows/Mac/Linux) [platform_count]",
    "Desteklenen dil sayısı [language_count]",
    "Fiyat (USD) [price_usd]",
    "Early Access mi? (1=Evet, 0=Hayır) [is_early_access]",
    "ChatGPT sonrası mı? (1=Aralık 2022+, 0=Öncesi) [post_chatgpt]",
    "Çıkış yılı (örn. 2023) [release_year]",
    "Steam Workshop var mı? (1=Evet, 0=Hayır) [has_workshop]",
]

# Target info (for regression)
TARGET_LABEL = "Tahmini İnceleme Skoru (Wilson Alt Sınırı)"
TARGET_UNIT  = "puan"
TARGET_MIN   = 0.0
TARGET_MAX   = 1.0

# Class labels (for classification — map integer outputs to names)
CLASS_LABELS = {}

# Model performance info shown in the UI
MODEL_INFO = {
    "model_name":    "Tuned Random Forest Regressor (Pipeline)",
    "metric1_label": "CV R²",
    "metric1_value": "0.1769",
    "metric2_label": "Test R²",
    "metric2_value": "0.1307",
    "training_note": "4.363 Steam indie oyunu üzerinde eğitildi. Hedef: Wilson alt sınırı (review_wilson_lower).",
}

# ================================================================
#  END OF STUDENT CONFIG
# ================================================================


# --- load model and scaler at startup ---
print(f"\n[demo_server] Loading model from: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print(f"[demo_server] Model loaded: {type(model).__name__}")
except FileNotFoundError:
    print(f"[demo_server] ERROR: Model file not found at '{MODEL_PATH}'")
    print("              Save your model first:  joblib.dump(model, 'model.joblib')")
    model = None

scaler = None
if SCALER_PATH:
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"[demo_server] Scaler loaded: {type(scaler).__name__}")
    except FileNotFoundError:
        print(f"[demo_server] WARNING: Scaler file not found at '{SCALER_PATH}'")


@app.route("/")
def index():
    """Serve the UI HTML file."""
    return send_from_directory(".", "demo_ui.html")


@app.route("/config")
def get_config():
    """Return model config so the UI can build itself dynamically."""
    return jsonify({
        "model_type":   MODEL_TYPE,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "target_label": TARGET_LABEL,
        "target_unit":  TARGET_UNIT,
        "target_min":   TARGET_MIN,
        "target_max":   TARGET_MAX,
        "class_labels": CLASS_LABELS,
        "model_info":   MODEL_INFO,
        "model_ready":  model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Receive feature values and return model prediction."""
    if model is None:
        return jsonify({"error": f"Model not loaded. Check that '{MODEL_PATH}' exists."}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Request must include a 'features' dict."}), 400

    raw = data["features"]

    # build feature vector in the correct order
    try:
        values = [float(raw[name]) for name in FEATURE_NAMES]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}. Expected: {FEATURE_NAMES}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    # reshape for sklearn
    X = np.array(values).reshape(1, -1)

    # optionally scale
    if scaler is not None:
        X = scaler.transform(X)

    # predict
    try:
        if MODEL_TYPE == "regression":
            pred = float(model.predict(X)[0])
            result = {
                "prediction": round(pred, 3),
                "target_label": TARGET_LABEL,
                "target_unit":  TARGET_UNIT,
                "target_min":   TARGET_MIN,
                "target_max":   TARGET_MAX,
            }

        else:  # classification
            pred_class = int(model.predict(X)[0])
            class_name = CLASS_LABELS.get(pred_class, str(pred_class))

            # confidence from predict_proba if available
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = round(float(max(proba)), 3)
                all_proba = {
                    CLASS_LABELS.get(i, str(i)): round(float(p), 3)
                    for i, p in enumerate(proba)
                }
            else:
                all_proba = {}

            result = {
                "predicted_class": class_name,
                "predicted_class_int": pred_class,
                "confidence": confidence,
                "all_probabilities": all_proba,
            }

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # optionally include feature importances if the model has them
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        result["feature_importances"] = {
            FEATURE_NAMES[i]: round(float(importances[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }
    elif hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        result["coefficients"] = {
            FEATURE_NAMES[i]: round(float(coefs[i]), 4)
            for i in range(min(len(FEATURE_NAMES), len(coefs)))
        }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ML Demo Server")
    print("  Open in browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
