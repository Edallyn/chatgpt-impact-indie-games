# ML Model Demo — Setup Guide
# Kurulum Kılavuzu

## Files / Dosyalar
  demo_server.py   ← Flask backend — loads your model, serves predictions
  demo_ui.html     ← Frontend UI — open in browser for the presentation

---

## Step 1 — Save your trained model
# Adım 1 — Modeli kaydet (Jupyter notebook'unuzda)

  import joblib
  joblib.dump(model, "model.joblib")             # your trained model
  joblib.dump(scaler, "scaler.joblib")           # only if you used a scaler

---

## Step 2 — Edit demo_server.py
# Adım 2 — demo_server.py'yi düzenle

  Open demo_server.py and fill in the STUDENT CONFIG section:

  MODEL_PATH      → path to your saved model file
  SCALER_PATH     → path to scaler, or None
  MODEL_TYPE      → "regression" or "classification"
  FEATURE_NAMES   → list of feature names (same order as model training!)
  FEATURE_LABELS  → human-readable labels shown in the UI
  TARGET_LABEL    → what the model predicts (e.g. "Predicted Price")
  TARGET_UNIT     → unit string (e.g. "₺/kg")
  TARGET_MIN/MAX  → expected output range
  CLASS_LABELS    → for classification: {0: "ClassName", 1: ...}
  MODEL_INFO      → model name and metrics to display in UI

---

## Step 3 — Install dependencies (once)
# Adım 3 — Bağımlılıkları kur (bir kez)

  pip install flask flask-cors joblib scikit-learn numpy pandas

---

## Step 4 — Run the server
# Adım 4 — Sunucuyu çalıştır

  python demo_server.py

  You should see:
    Model loaded: RandomForestRegressor
    Open in browser: http://localhost:5000

---

## Step 5 — Open the demo
# Adım 5 — Demo'yu aç

  Open your browser and go to:  http://localhost:5000
  The green dot in the bottom bar confirms the model is connected.

---

## Troubleshooting / Sorun Giderme

  "Model file not found"
    → Make sure model.joblib is in the same folder as demo_server.py
    → Or update MODEL_PATH to the correct path

  "Cannot connect to localhost:5000"
    → Make sure demo_server.py is running
    → Check the terminal for error messages

  "Missing feature" error
    → FEATURE_NAMES must match exactly what was used during model.fit()
    → Check column order in your training DataFrame

  Prediction looks wrong
    → If you used a scaler during training, set SCALER_PATH and save the scaler too
    → Make sure categorical features are encoded the same way as during training
    → FEATURE_NAMES should match the column names that you used during model.fit()
