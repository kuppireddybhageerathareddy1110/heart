#!/bin/bash
# ================================================================
# HEART DISEASE PREDICTION - QUICK START SCRIPT (Mac/Linux)
# ================================================================

echo ""
echo "========================================"
echo "HEART DISEASE AI - QUICK START"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet fastapi "uvicorn[standard]" pydantic
pip install --quiet numpy pandas matplotlib seaborn
pip install --quiet scikit-learn xgboost lightgbm tensorflow
pip install --quiet shap lime joblib

# Check if model files exist
if [ ! -f "xgboost_model.pkl" ] && [ ! -f "outputs/xgboost_model.pkl" ]; then
    echo ""
    echo "==============================================="
    echo "ERROR: Model files not found!"
    echo "==============================================="
    echo ""
    echo "Please run the training script first:"
    echo "  python heart_disease_prediction_complete.py"
    echo ""
    echo "Or ensure the following files exist:"
    echo "  - xgboost_model.pkl"
    echo "  - xgboost_scaler.pkl"
    echo "  - xgboost_features.json"
    echo ""
    exit 1
fi

# Start API server
echo ""
echo "========================================"
echo "Starting API Server..."
echo "========================================"
echo ""
echo "API will be available at: http://localhost:8001"
echo "API Documentation: http://localhost:8001/docs"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

python -m uvicorn api_integrated:app --host 0.0.0.0 --port 8001 --reload
