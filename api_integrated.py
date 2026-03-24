"""
================================================================================
HEART DISEASE PREDICTION API - PRODUCTION DEPLOYMENT
Senior Design Project
Author: [Your Name]
Date: March 2026

FastAPI backend for Heart Disease Prediction with Explainable AI
Works with models trained by heart_disease_prediction_complete.py

Required Files (in outputs/ or same directory):
  - xgboost_model.pkl
  - xgboost_scaler.pkl  
  - xgboost_features.json

Run Command:
  uvicorn api_integrated:app --host 0.0.0.0 --port 8001 --reload

API Documentation:
  http://localhost:8001/docs
================================================================================
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
import seaborn as sns

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from api_integrated import app
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent

# Try to load from outputs/ directory first (if running from project root)
# Otherwise load from same directory as api.py
if (BASE_DIR / "outputs").exists():
    MODEL_DIR = BASE_DIR / "outputs"
else:
    MODEL_DIR = BASE_DIR

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
SCALER_PATH = MODEL_DIR / "xgboost_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "xgboost_features.json"

# ==========================================
# LIFESPAN HANDLER
# ==========================================

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Startup and shutdown events"""
    logger.info("=" * 80)
    logger.info("🚀 Heart Disease Prediction API - Starting Up")
    logger.info("=" * 80)
    logger.info(f"Model Path:    {MODEL_PATH}")
    logger.info(f"Scaler Path:   {SCALER_PATH}")
    logger.info(f"Features Path: {FEATURES_PATH}")
    logger.info(f"Model Type:    {type(model).__name__}")
    logger.info(f"Features:      {len(feature_names)}")
    logger.info("=" * 80)
    yield
    logger.info("👋 Heart Disease Prediction API - Shutting Down")

# ==========================================
# FASTAPI APP INITIALIZATION
# ==========================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-grade heart disease prediction with Explainable AI using XGBoost",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD MODEL AND ARTIFACTS
# ==========================================

try:
    logger.info("Loading model artifacts...")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        feature_data = json.load(f)
        if isinstance(feature_data, dict):
            feature_names = feature_data.get('features', feature_data)
        else:
            feature_names = feature_data
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    logger.info("✅ Model artifacts loaded successfully")
    logger.info(f"   Model type: {type(model).__name__}")
    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Feature list: {feature_names[:5]}... (showing first 5)")
    
except FileNotFoundError as e:
    logger.error(f"❌ Missing model file: {e}")
    logger.error(f"   Please ensure the following files exist:")
    logger.error(f"   - {MODEL_PATH}")
    logger.error(f"   - {SCALER_PATH}")
    logger.error(f"   - {FEATURES_PATH}")
    raise
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# ==========================================
# PYDANTIC MODELS (REQUEST/RESPONSE)
# ==========================================

class PatientInput(BaseModel):
    """Patient data input schema with validation"""
    
    age: float = Field(..., ge=18, le=120, description="Age in years")
    sex: str = Field(..., description="Sex: Male or Female")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression (Oldpeak)")
    chest_pain: str = Field(..., description="Chest pain type")
    restingbp_final: float = Field(..., ge=80, le=200, description="Resting BP (mm Hg)")
    chol_final: float = Field(..., ge=100, le=600, description="Cholesterol (mg/dl)")
    maxhr_final: float = Field(..., ge=60, le=220, description="Maximum heart rate")
    fasting_bs: str = Field(..., description="Fasting blood sugar")
    resting_ecg: str = Field(..., description="Resting ECG result")
    exercise_angina: str = Field(..., description="Exercise-induced angina")
    st_slope: str = Field(..., description="ST slope")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Sex must be Male or Female')
        return v
    
    @field_validator('chest_pain')
    @classmethod
    def validate_chest_pain(cls, v):
        valid = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        if v not in valid:
            raise ValueError(f'Chest pain must be one of: {valid}')
        return v
    
    @field_validator('fasting_bs')
    @classmethod
    def validate_fasting_bs(cls, v):
        if v not in ['No', 'Yes']:
            raise ValueError('Fasting BS must be No or Yes')
        return v
    
    @field_validator('resting_ecg')
    @classmethod
    def validate_resting_ecg(cls, v):
        valid = ['Normal', 'ST-T Abnormality', 'LV Hypertrophy']
        if v not in valid:
            raise ValueError(f'Resting ECG must be one of: {valid}')
        return v
    
    @field_validator('exercise_angina')
    @classmethod
    def validate_exercise_angina(cls, v):
        if v not in ['No', 'Yes']:
            raise ValueError('Exercise angina must be No or Yes')
        return v
    
    @field_validator('st_slope')
    @classmethod
    def validate_st_slope(cls, v):
        valid = ['Upsloping', 'Flat', 'Downsloping']
        if v not in valid:
            raise ValueError(f'ST slope must be one of: {valid}')
        return v


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    prediction: str
    probability: float
    confidence_interval: Dict[str, float]
    risk_level: str
    risk_factors: List[str]
    protective_factors: List[str]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientInput]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    features_count: int
    timestamp: str


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def encode_patient_data(patient: PatientInput) -> pd.DataFrame:
    """
    Encode patient data into the format expected by the model.
    Creates a DataFrame with all feature columns in the correct order.
    """
    # Initialize with zeros
    data = {feature: 0.0 for feature in feature_names}
    
    # Set numeric features
    numeric_features = ['age', 'oldpeak', 'restingbp_final', 'chol_final', 'maxhr_final']
    for feat in numeric_features:
        if feat in data:
            data[feat] = float(getattr(patient, feat))
    
    # Encode sex (Male = 1)
    if 'sex_M' in data:
        data['sex_M'] = 1.0 if patient.sex == 'Male' else 0.0
    
    # Encode chest pain (one-hot encoding, reference category is Typical Angina = 0)
    cp_mapping = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }
    cp_value = cp_mapping[patient.chest_pain]
    for i in range(1, 4):
        key = f'cp_final_{i}'
        if key in data:
            data[key] = 1.0 if i == cp_value else 0.0
    
    # Encode fasting blood sugar
    if 'fbs_final_Yes' in data:
        data['fbs_final_Yes'] = 1.0 if patient.fasting_bs == 'Yes' else 0.0
    
    # Encode resting ECG (reference category is Normal = 0)
    ecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 'LV Hypertrophy': 2}
    ecg_value = ecg_mapping[patient.resting_ecg]
    for i in range(1, 3):
        key = f'restecg_final_{i}'
        if key in data:
            data[key] = 1.0 if i == ecg_value else 0.0
    
    # Encode exercise angina
    if 'exang_final_Y' in data:
        data['exang_final_Y'] = 1.0 if patient.exercise_angina == 'Yes' else 0.0
    
    # Encode ST slope (reference category is Upsloping = 0)
    slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope_value = slope_mapping[patient.st_slope]
    for i in range(1, 3):
        key = f'slope_final_{i}'
        if key in data:
            data[key] = 1.0 if i == slope_value else 0.0
    
    return pd.DataFrame([data])[feature_names]


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Scale features using the loaded scaler"""
    return scaler.transform(df).astype(np.float64)


def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def get_confidence_interval(probability: float) -> Dict[str, float]:
    """Calculate confidence interval (simplified)"""
    margin = 0.05  # ±5% confidence interval
    return {
        "lower": round(max(0.0, probability - margin), 4),
        "upper": round(min(1.0, probability + margin), 4),
        "confidence_level": 0.95
    }


def extract_shap_values(shap_raw) -> np.ndarray:
    """
    Extract SHAP values for the disease class (class 1) from various formats.
    Handles different SHAP output formats for binary classification.
    """
    if isinstance(shap_raw, list):
        # Format: [class_0_array, class_1_array]
        return np.array(shap_raw[1]).flatten()
    
    arr = np.array(shap_raw)
    
    if arr.ndim == 3:
        # Format: (n_samples, n_features, n_classes)
        return arr[0, :, 1]
    
    if arr.ndim == 2:
        # Format: (n_samples, n_features)
        return arr[0]
    
    return arr.flatten()


def get_base_value() -> float:
    """Get SHAP base value (expected value)"""
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])  # Disease class
    return float(ev)


def identify_risk_factors(shap_vals: np.ndarray, patient_df: pd.DataFrame):
    """
    Identify top risk and protective factors based on SHAP values.
    
    Returns:
        Tuple of (risk_factors, protective_factors)
    """
    # Combine feature names, SHAP values, and actual values
    feature_impacts = list(zip(
        feature_names,
        shap_vals,
        patient_df.values[0]
    ))
    
    # Sort by absolute SHAP value (importance)
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top risk factors (positive SHAP values)
    risk_factors = [
        f"{feat} ({val:.2f})"
        for feat, shap_val, val in feature_impacts[:6]
        if shap_val > 0
    ]
    
    # Get top protective factors (negative SHAP values)
    protective_factors = [
        f"{feat} ({val:.2f})"
        for feat, shap_val, val in feature_impacts[:6]
        if shap_val < 0
    ]
    
    return risk_factors, protective_factors


def figure_to_base64() -> str:
    """Convert current matplotlib figure to base64 string"""
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close('all')
    return b64


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "3.0.0",
        "status": "running",
        "model": type(model).__name__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=type(model).__name__,
        features_count=len(feature_names),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """
    Single patient prediction with risk assessment.
    
    Returns:
        - Prediction (Disease / No Disease)
        - Probability score
        - Confidence interval
        - Risk level (LOW / MEDIUM / HIGH)
        - Risk factors
        - Protective factors
    """
    try:
        logger.info(f"Prediction request for patient age {patient.age}, sex {patient.sex}")
        
        # Encode patient data
        patient_df = encode_patient_data(patient)
        
        # Scale features
        patient_scaled = scale_features(patient_df)
        
        # Make prediction
        prediction = int(model.predict(patient_scaled)[0])
        probabilities = model.predict_proba(patient_scaled)[0]
        disease_prob = float(probabilities[1])
        
        # Get SHAP values for explanation
        shap_raw = explainer.shap_values(patient_df)
        shap_vals = extract_shap_values(shap_raw)
        
        # Identify risk and protective factors
        risk_factors, protective_factors = identify_risk_factors(shap_vals, patient_df)
        
        # Build response
        response = PredictionResponse(
            prediction="Disease" if prediction == 1 else "No Disease",
            probability=disease_prob,
            confidence_interval=get_confidence_interval(disease_prob),
            risk_level=get_risk_level(disease_prob),
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed: {response.prediction} ({disease_prob:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain/waterfall")
async def explain_waterfall(patient: PatientInput):
    """
    Generate SHAP waterfall plot for individual prediction.
    Shows how each feature contributes to pushing the prediction higher or lower.
    """
    try:
        patient_df = encode_patient_data(patient)
        shap_raw = explainer.shap_values(patient_df)
        vals = extract_shap_values(shap_raw)
        base = get_base_value()
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=vals,
            base_values=base,
            data=patient_df.iloc[0].values,
            feature_names=feature_names
        )
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title("SHAP Waterfall Plot - Feature Contributions", 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return {"plot": figure_to_base64(), "type": "waterfall"}
        
    except Exception as e:
        logger.error(f"Waterfall plot error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/force")
async def explain_force(patient: PatientInput):
    """
    Generate SHAP force plot (interactive visualization).
    Shows cumulative impact of features pushing prediction higher or lower.
    """
    try:
        patient_df = encode_patient_data(patient)
        shap_raw = explainer.shap_values(patient_df)
        vals = extract_shap_values(shap_raw)
        base = get_base_value()
        
        # Create force plot
        force_plot = shap.force_plot(
            base,  # base_value
            vals,  # shap_values
            patient_df.iloc[0],  # features
            feature_names=feature_names
        )
        
        # Save to HTML
        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        
        return {"plot": html_buffer.getvalue(), "type": "force"}
        
    except Exception as e:
        logger.error(f"Force plot error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/summary")
async def explain_summary():
    """
    Generate global SHAP summary plot.
    Shows overall feature importance across all predictions.
    Uses synthetic data to avoid requiring the full training dataset.
    """
    try:
        # Generate synthetic data for summary
        rng = np.random.default_rng(42)
        n_samples = 100
        synth_data = pd.DataFrame(
            rng.standard_normal((n_samples, len(feature_names))),
            columns=feature_names
        )
        
        # Clip binary features to {0, 1}
        numeric_features = ['age', 'oldpeak', 'restingbp_final', 'chol_final', 'maxhr_final']
        for col in feature_names:
            if col not in numeric_features:
                synth_data[col] = (synth_data[col] > 0).astype(float)
        
        # Get SHAP values
        shap_raw = explainer.shap_values(synth_data)
        
        if isinstance(shap_raw, list):
            vals = shap_raw[1]  # Disease class
        elif np.array(shap_raw).ndim == 3:
            vals = np.array(shap_raw)[:, :, 1]
        else:
            vals = np.array(shap_raw)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(vals, synth_data, show=False, max_display=15)
        plt.title("Global Feature Importance - SHAP Summary", 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return {"plot": figure_to_base64(), "type": "summary"}
        
    except Exception as e:
        logger.error(f"Summary plot error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/importance")
async def explain_importance():
    """
    Generate feature importance bar plot.
    Shows which features are most important overall (mean absolute SHAP values).
    """
    try:
        # Generate synthetic data
        rng = np.random.default_rng(42)
        n_samples = 100
        synth_data = pd.DataFrame(
            rng.standard_normal((n_samples, len(feature_names))),
            columns=feature_names
        )
        
        numeric_features = ['age', 'oldpeak', 'restingbp_final', 'chol_final', 'maxhr_final']
        for col in feature_names:
            if col not in numeric_features:
                synth_data[col] = (synth_data[col] > 0).astype(float)
        
        shap_raw = explainer.shap_values(synth_data)
        
        if isinstance(shap_raw, list):
            vals = shap_raw[1]
        elif np.array(shap_raw).ndim == 3:
            vals = np.array(shap_raw)[:, :, 1]
        else:
            vals = np.array(shap_raw)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(vals, synth_data, plot_type="bar", show=False, max_display=15)
        plt.title("Feature Importance - Mean Absolute SHAP Values", 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return {"plot": figure_to_base64(), "type": "importance"}
        
    except Exception as e:
        logger.error(f"Importance plot error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime")
async def explain_lime(patient: PatientInput):
    """
    Generate LIME explanation for individual prediction.
    LIME creates a simple interpretable model around the specific prediction.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        patient_df = encode_patient_data(patient)
        patient_scaled = scale_features(patient_df)
        
        # Generate synthetic training data for LIME
        rng = np.random.default_rng(42)
        n_samples = 100
        synth_data = pd.DataFrame(
            rng.standard_normal((n_samples, len(feature_names))),
            columns=feature_names
        )
        
        numeric_features = ['age', 'oldpeak', 'restingbp_final', 'chol_final', 'maxhr_final']
        for col in feature_names:
            if col not in numeric_features:
                synth_data[col] = (synth_data[col] > 0).astype(float)
        
        training_scaled = scaler.transform(synth_data).astype(np.float64)
        
        # Create LIME explainer
        lime_explainer = LimeTabularExplainer(
            training_data=training_scaled,
            feature_names=feature_names,
            class_names=['No Disease', 'Disease'],
            mode='classification',
            random_state=42
        )
        
        # Generate explanation
        exp = lime_explainer.explain_instance(
            patient_scaled[0],
            model.predict_proba,
            num_features=12
        )
        
        # Create plot
        fig = exp.as_pyplot_figure()
        plt.title("LIME Explanation - Top Contributing Features", 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return {
            "plot": figure_to_base64(),
            "type": "lime",
            "feature_weights": [
                {"feature": str(feat), "weight": float(weight)}
                for feat, weight in exp.as_list()
            ]
        }
        
    except Exception as e:
        logger.error(f"LIME error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple patients.
    Efficiently processes multiple patients and returns all predictions.
    """
    try:
        results = []
        
        for patient in request.patients:
            patient_df = encode_patient_data(patient)
            patient_scaled = scale_features(patient_df)
            
            prediction = int(model.predict(patient_scaled)[0])
            probabilities = model.predict_proba(patient_scaled)[0]
            disease_prob = float(probabilities[1])
            
            results.append({
                "age": patient.age,
                "sex": patient.sex,
                "prediction": "Disease" if prediction == 1 else "No Disease",
                "probability": disease_prob,
                "risk_level": get_risk_level(disease_prob)
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        
        return {
            "total_patients": len(results),
            "predictions": results,
            "summary": {
                "disease_count": sum(1 for r in results if r["prediction"] == "Disease"),
                "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get detailed model information and metadata"""
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "features": feature_names,
        "needs_scaling": True,
        "algorithm": "XGBoost Classifier",
        "training_info": {
            "dataset": "Heart Disease Dataset",
            "validation": "5-fold cross-validation",
            "random_state": 42
        },
        "performance_note": "See outputs/21_complete_model_comparison.csv for detailed metrics"
    }


@app.get("/features/list")
async def list_features():
    """Get list of all features used by the model"""
    numeric = ['age', 'oldpeak', 'restingbp_final', 'chol_final', 'maxhr_final']
    categorical = [f for f in feature_names if f not in numeric]
    
    return {
        "features": feature_names,
        "count": len(feature_names),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "encoding_info": {
            "sex": "One-hot (sex_M)",
            "chest_pain": "One-hot (cp_final_1, cp_final_2, cp_final_3)",
            "fasting_bs": "Binary (fbs_final_Yes)",
            "resting_ecg": "One-hot (restecg_final_1, restecg_final_2)",
            "exercise_angina": "Binary (exang_final_Y)",
            "st_slope": "One-hot (slope_final_1, slope_final_2)"
        }
    }


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("HEART DISEASE PREDICTION API")
    print("=" * 80)
    print(f"Model: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Starting server on http://0.0.0.0:8001")
    print(f"API Docs: http://0.0.0.0:8001/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        "api_integrated:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
