# ==========================================
# HEART DISEASE PREDICTION - FINAL VERSION
# (XGBOOST DEPLOYMENT - CLEAN & CORRECT)
# ==========================================

import joblib
import numpy as np
import pandas as pd

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "xgboost_scaler.pkl"

print("📦 Loading model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model loaded successfully")

# ==========================================
# PREPARE INPUT FUNCTION
# ==========================================

def prepare_input(patient):

    data = {}

    # 🔢 Numerical features
    data["age"] = patient["age"]
    data["oldpeak"] = patient["oldpeak"]
    data["restingbp_final"] = patient["restingbp_final"]
    data["chol_final"] = patient["chol_final"]
    data["maxhr_final"] = patient["maxhr_final"]

    # 👤 Sex
    data["sex_M"] = 1 if patient["sex"] == "Male" else 0

    # 🫀 Chest Pain
    data["cp_final_ATA"] = 1 if patient["cp"] == "ATA" else 0
    data["cp_final_NAP"] = 1 if patient["cp"] == "NAP" else 0
    data["cp_final_TA"] = 1 if patient["cp"] == "TA" else 0

    # 🧪 Fasting Blood Sugar
    data["fbs_final_Yes"] = 1 if patient["fbs"] == "Yes" else 0

    # 📉 ECG
    data["restecg_final_Normal"] = 1 if patient["restecg"] == "Normal" else 0
    data["restecg_final_ST"] = 1 if patient["restecg"] == "ST" else 0

    # 🏃 Exercise Angina
    data["exang_final_Y"] = 1 if patient["exang"] == "Yes" else 0

    # 📊 ST Slope
    data["slope_final_Flat"] = 1 if patient["slope"] == "Flat" else 0
    data["slope_final_Up"] = 1 if patient["slope"] == "Up" else 0

    # ==========================================
    # ENGINEERED FEATURES (MUST MATCH TRAINING)
    # ==========================================

    data["chol_age_ratio"] = data["chol_final"] / data["age"]
    data["bp_hr_ratio"] = data["restingbp_final"] / data["maxhr_final"]
    data["stress_index"] = data["oldpeak"] * data["maxhr_final"]
    data["cardiac_load"] = data["restingbp_final"] * data["age"]

    # Convert to DataFrame
    df = pd.DataFrame([data])

    return df

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict(patient):

    # Prepare input
    X = prepare_input(patient)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    pred = model.predict(X_scaled)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        prob = float(pred)

    return pred, prob

# ==========================================
# TEST RUN
# ==========================================

if __name__ == "__main__":

    sample_patient = {
        "age": 55,
        "sex": "Male",
        "oldpeak": 2.0,
        "cp": "ATA",
        "restingbp_final": 140,
        "chol_final": 250,
        "maxhr_final": 150,
        "fbs": "No",
        "restecg": "Normal",
        "exang": "No",
        "slope": "Flat"
    }

    prediction, probability = predict(sample_patient)

    print("\n🧠 Prediction Result:")
    print(f"Prediction: {'Disease' if prediction == 1 else 'No Disease'}")
    print(f"Probability: {probability:.2%}")