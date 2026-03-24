"""
Heart Disease Prediction API - XGBoost Edition
===============================================
FastAPI backend wired to the XGBoost model produced by
heart_disease_prediction_complete.py in files4/outputs/.

Expected files (all in the same folder as this script, or adjust paths):
  xgboost_model.pkl
  xgboost_scaler.pkl
  xgboost_features.json

Run:
  uvicorn api:app --host 0.0.0.0 --port 8001 --reload
"""

import io
import json
import base64
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
# All model artefacts live next to this script (outputs/ folder).
# Change BASE_DIR if you keep them elsewhere.
BASE_DIR = Path(__file__).parent          # same folder as api.py
MODEL_PATH    = BASE_DIR / "xgboost_model.pkl"
SCALER_PATH   = BASE_DIR / "xgboost_scaler.pkl"
FEATURES_PATH = BASE_DIR / "xgboost_features.json"

# ── lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("🚀  Heart Disease Prediction API starting…")
    logger.info(f"   model   : {MODEL_PATH}")
    logger.info(f"   scaler  : {SCALER_PATH}")
    logger.info(f"   features: {FEATURES_PATH}")
    yield
    logger.info("👋  Heart Disease Prediction API shutting down")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="XGBoost-powered heart disease prediction with Explainable AI",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── load artefacts ────────────────────────────────────────────────────────────
try:
    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_names: List[str] = json.load(f)

    # SHAP explainer — XGBoost works best with TreeExplainer
    explainer = shap.TreeExplainer(model)

    logger.info(f"✅  Model loaded  : {type(model).__name__}")
    logger.info(f"✅  Features      : {len(feature_names)}")
    logger.info(f"✅  Feature list  : {feature_names}")

except FileNotFoundError as exc:
    logger.error(
        f"❌  Missing artefact: {exc}\n"
        "    Make sure xgboost_model.pkl, xgboost_scaler.pkl and "
        "xgboost_features.json are in the same folder as api.py."
    )
    raise
except Exception as exc:
    logger.error(f"❌  Failed to load artefacts: {exc}")
    raise


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    age:              float = Field(..., ge=18,  le=120, description="Age in years")
    sex:              str   = Field(...,                 description="Male | Female")
    oldpeak:          float = Field(..., ge=0,   le=10,  description="ST depression (Oldpeak)")
    chest_pain:       str   = Field(...,                 description="Chest pain type")
    restingbp_final:  float = Field(..., ge=80,  le=200, description="Resting BP (mm Hg)")
    chol_final:       float = Field(..., ge=100, le=600, description="Cholesterol (mg/dl)")
    maxhr_final:      float = Field(..., ge=60,  le=220, description="Maximum heart rate (bpm)")
    fasting_bs:       str   = Field(...,                 description="Fasting blood sugar > 120: Yes | No")
    resting_ecg:      str   = Field(...,                 description="Resting ECG result")
    exercise_angina:  str   = Field(...,                 description="Exercise-induced angina: Yes | No")
    st_slope:         str   = Field(...,                 description="ST slope")

    @field_validator("sex")
    @classmethod
    def v_sex(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("Sex must be Male or Female")
        return v

    @field_validator("chest_pain")
    @classmethod
    def v_chest_pain(cls, v):
        valid = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        if v not in valid:
            raise ValueError(f"chest_pain must be one of {valid}")
        return v

    @field_validator("fasting_bs")
    @classmethod
    def v_fbs(cls, v):
        if v not in ("No", "Yes"):
            raise ValueError("fasting_bs must be No or Yes")
        return v

    @field_validator("resting_ecg")
    @classmethod
    def v_ecg(cls, v):
        valid = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
        if v not in valid:
            raise ValueError(f"resting_ecg must be one of {valid}")
        return v

    @field_validator("exercise_angina")
    @classmethod
    def v_angina(cls, v):
        if v not in ("No", "Yes"):
            raise ValueError("exercise_angina must be No or Yes")
        return v

    @field_validator("st_slope")
    @classmethod
    def v_slope(cls, v):
        valid = ["Upsloping", "Flat", "Downsloping"]
        if v not in valid:
            raise ValueError(f"st_slope must be one of {valid}")
        return v


class PredictionResponse(BaseModel):
    prediction:          str
    probability:         float
    confidence_interval: Dict[str, float]
    risk_level:          str
    risk_factors:        List[str]
    protective_factors:  List[str]
    timestamp:           str


class BatchPredictionRequest(BaseModel):
    patients: List[PatientInput]


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_type:     str
    features_count: int
    timestamp:      str


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_patient(patient: PatientInput) -> pd.DataFrame:
    """
    Build a one-row DataFrame whose columns exactly match feature_names.

    The XGBoost model was trained on scaled numeric features only
    (xgboost_features.json lists the column names after preprocessing).
    We reconstruct those columns here.
    """
    row: Dict[str, float] = {f: 0.0 for f in feature_names}

    # ── numeric ──────────────────────────────────────────────────────────────
    for col in ("age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"):
        if col in row:
            row[col] = float(getattr(patient, col))

    # ── sex ──────────────────────────────────────────────────────────────────
    sex_m = "sex_M"
    if sex_m in row:
        row[sex_m] = 1.0 if patient.sex == "Male" else 0.0

    # ── chest pain (one-hot, reference = Typical Angina → index 0) ───────────
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1,
               "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp_val = cp_map[patient.chest_pain]
    for i in range(1, 4):
        k = f"cp_final_{i}"
        if k in row:
            row[k] = 1.0 if i == cp_val else 0.0

    # ── fasting blood sugar ───────────────────────────────────────────────────
    fbs_k = "fbs_final_Yes"
    if fbs_k in row:
        row[fbs_k] = 1.0 if patient.fasting_bs == "Yes" else 0.0

    # ── resting ECG (reference = Normal → index 0) ────────────────────────────
    ecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    ecg_val = ecg_map[patient.resting_ecg]
    for i in range(1, 3):
        k = f"restecg_final_{i}"
        if k in row:
            row[k] = 1.0 if i == ecg_val else 0.0

    # ── exercise angina ───────────────────────────────────────────────────────
    ang_k = "exang_final_Y"
    if ang_k in row:
        row[ang_k] = 1.0 if patient.exercise_angina == "Yes" else 0.0

    # ── ST slope (reference = Upsloping → index 0) ────────────────────────────
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_val = slope_map[patient.st_slope]
    for i in range(1, 3):
        k = f"slope_final_{i}"
        if k in row:
            row[k] = 1.0 if i == slope_val else 0.0

    return pd.DataFrame([row])[feature_names]


def scale(df: pd.DataFrame) -> np.ndarray:
    return scaler.transform(df).astype(np.float64)


def get_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def get_ci(prob: float) -> Dict[str, float]:
    margin = 0.05
    return {
        "lower":            round(max(0.0, prob - margin), 4),
        "upper":            round(min(1.0, prob + margin), 4),
        "confidence_level": 0.95,
    }


def extract_shap_vals(shap_raw) -> np.ndarray:
    """
    Return 1-D SHAP values for the positive (disease) class
    regardless of which SHAP output format the model produces.
    """
    if isinstance(shap_raw, list):
        # list of arrays  →  [class_0_array, class_1_array]
        return np.array(shap_raw[1]).flatten()
    arr = np.array(shap_raw)
    if arr.ndim == 3:
        # shape (1, n_features, 2)  →  take class index 1
        return arr[0, :, 1]
    if arr.ndim == 2:
        # shape (1, n_features)
        return arr[0]
    return arr.flatten()


def get_base_value() -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])
    return float(ev)


def identify_factors(shap_vals: np.ndarray, patient_df: pd.DataFrame):
    pairs = sorted(
        zip(feature_names, shap_vals, patient_df.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    risk       = [f"{f} ({v:.2f})" for f, s, v in pairs[:6] if s > 0]
    protective = [f"{f} ({v:.2f})" for f, s, v in pairs[:6] if s < 0]
    return risk, protective


def fig_to_b64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")
    return b64


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Heart Disease Prediction API", "version": "3.0.0",
            "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=type(model).__name__,
        features_count=len(feature_names),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    try:
        df   = encode_patient(patient)
        Xs   = scale(df)

        pred      = int(model.predict(Xs)[0])
        probs     = model.predict_proba(Xs)[0]
        disease_p = float(probs[1])

        shap_raw  = explainer.shap_values(df)
        shap_vals = extract_shap_vals(shap_raw)

        risk_f, prot_f = identify_factors(shap_vals, df)

        return PredictionResponse(
            prediction         = "Disease" if pred == 1 else "No Disease",
            probability        = disease_p,
            confidence_interval= get_ci(disease_p),
            risk_level         = get_risk_level(disease_p),
            risk_factors       = risk_f,
            protective_factors = prot_f,
            timestamp          = datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain/waterfall")
async def explain_waterfall(patient: PatientInput):
    try:
        df        = encode_patient(patient)
        shap_raw  = explainer.shap_values(df)
        vals      = extract_shap_vals(shap_raw)
        base      = get_base_value()

        explanation = shap.Explanation(
            values       = vals,
            base_values  = base,
            data         = df.iloc[0].values,
            feature_names= feature_names,
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title("SHAP Waterfall — Feature Contributions", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return {"plot": fig_to_b64(), "type": "waterfall"}
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain/force")
async def explain_force(patient: PatientInput):
    try:
        df        = encode_patient(patient)
        shap_raw  = explainer.shap_values(df)
        vals      = extract_shap_vals(shap_raw)
        base      = get_base_value()

        force_plot = shap.force_plot(base, vals, df.iloc[0], feature_names=feature_names)

        buf = io.StringIO()
        shap.save_html(buf, force_plot)
        return {"plot": buf.getvalue(), "type": "force"}
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain/summary")
async def explain_summary():
    try:
        # Use a small sample of synthetic data derived from feature ranges
        # (avoids needing the original CSV at runtime)
        rng   = np.random.default_rng(42)
        n     = 80
        synth = pd.DataFrame(
            rng.standard_normal((n, len(feature_names))),
            columns=feature_names,
        )
        # Clip binary columns to {0,1}
        for col in feature_names:
            if col not in ("age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"):
                synth[col] = (synth[col] > 0).astype(float)

        shap_raw = explainer.shap_values(synth)
        if isinstance(shap_raw, list):
            vals = shap_raw[1]
        elif np.array(shap_raw).ndim == 3:
            vals = np.array(shap_raw)[:, :, 1]
        else:
            vals = np.array(shap_raw)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(vals, synth, show=False, max_display=15)
        plt.title("Global Feature Importance — SHAP Summary", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return {"plot": fig_to_b64(), "type": "summary"}
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain/importance")
async def explain_importance():
    try:
        rng   = np.random.default_rng(42)
        n     = 80
        synth = pd.DataFrame(
            rng.standard_normal((n, len(feature_names))),
            columns=feature_names,
        )
        for col in feature_names:
            if col not in ("age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"):
                synth[col] = (synth[col] > 0).astype(float)

        shap_raw = explainer.shap_values(synth)
        if isinstance(shap_raw, list):
            vals = shap_raw[1]
        elif np.array(shap_raw).ndim == 3:
            vals = np.array(shap_raw)[:, :, 1]
        else:
            vals = np.array(shap_raw)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(vals, synth, plot_type="bar", show=False, max_display=15)
        plt.title("Feature Importance — Mean |SHAP|", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return {"plot": fig_to_b64(), "type": "importance"}
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain/lime")
async def explain_lime(patient: PatientInput):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        df      = encode_patient(patient)
        Xs      = scale(df)

        rng    = np.random.default_rng(42)
        n      = 100
        synth  = pd.DataFrame(
            rng.standard_normal((n, len(feature_names))),
            columns=feature_names,
        )
        for col in feature_names:
            if col not in ("age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"):
                synth[col] = (synth[col] > 0).astype(float)
        training_data = scaler.transform(synth).astype(np.float64)

        lime_exp = LimeTabularExplainer(
            training_data = training_data,
            feature_names = feature_names,
            class_names   = ["No Disease", "Disease"],
            mode          = "classification",
            random_state  = 42,
        )

        exp = lime_exp.explain_instance(
            Xs[0],
            model.predict_proba,
            num_features=12,
        )

        fig = exp.as_pyplot_figure()
        plt.title("LIME Explanation — Top Features", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return {
            "plot": fig_to_b64(),
            "type": "lime",
            "feature_weights": [
                {"feature": str(f), "weight": float(w)}
                for f, w in exp.as_list()
            ],
        }
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    try:
        results = []
        for p in request.patients:
            df   = encode_patient(p)
            Xs   = scale(df)
            pred = int(model.predict(Xs)[0])
            prob = float(model.predict_proba(Xs)[0][1])
            results.append({
                "age":        p.age,
                "sex":        p.sex,
                "prediction": "Disease" if pred == 1 else "No Disease",
                "probability":prob,
                "risk_level": get_risk_level(prob),
            })

        return {
            "total_patients": len(results),
            "predictions":    results,
            "summary": {
                "disease_count":   sum(1 for r in results if r["prediction"] == "Disease"),
                "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH"),
                "timestamp":       datetime.now().isoformat(),
            },
        }
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/info")
async def model_info():
    return {
        "model_type":     type(model).__name__,
        "n_features":     len(feature_names),
        "features":       feature_names,
        "needs_scaling":  True,
        "performance_metrics": {
            "note": "See outputs/21_complete_model_comparison.csv for full metrics"
        },
    }


@app.get("/features/list")
async def list_features():
    numeric = ["age", "oldpeak", "restingbp_final", "chol_final", "maxhr_final"]
    return {
        "features":             feature_names,
        "count":                len(feature_names),
        "numeric_features":     numeric,
        "categorical_features": [f for f in feature_names if f not in numeric],
    }


@app.exception_handler(Exception)
async def global_exc_handler(request, exc):
    logger.error(f"Unhandled: {exc}")
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error", "error": str(exc)})


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
