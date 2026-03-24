# 🫀 Heart Disease Prediction System - Complete Deployment Package

## Senior Design Project - Explainable AI for Heart Disease Prediction

---

## 🎯 Quick Start (3 Steps)

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Start Backend API**
```bash
# Windows
start_api.bat

# Mac/Linux
chmod +x start_api.sh
./start_api.sh
```

### **3. Open Web Interface**
- Double-click `index.html`
- Or visit `http://localhost:8000` if using HTTP server

**That's it!** The system is now running.

---

## 📦 What's Included

### **Complete ML/DL Pipeline**
- ✅ 11 Traditional ML Models (Logistic Regression, Decision Trees, Random Forest, XGBoost, etc.)
- ✅ 7 Deep Learning Architectures (ANN, Deep ANN, Dropout, BatchNorm, Wide-Deep, ResNet)
- ✅ 4 Hybrid Ensemble Models (Averaging, Weighted, Stacking, Voting)
- ✅ Automatic model selection and deployment

### **Explainable AI**
- ✅ SHAP Analysis (Waterfall, Force, Summary, Dependence, Interactions)
- ✅ LIME Explanations (Local interpretability)
- ✅ Partial Dependence Plots
- ✅ Feature Importance Rankings
- ✅ Feature Correlation Analysis

### **Production-Ready API**
- ✅ FastAPI backend with automatic documentation
- ✅ Input validation and error handling
- ✅ Batch prediction support
- ✅ Comprehensive logging
- ✅ CORS enabled for web access

### **Modern Web Interface**
- ✅ Interactive patient data entry
- ✅ Real-time risk assessment
- ✅ Visual probability gauge
- ✅ Risk and protective factors display
- ✅ Multiple explanation visualizations
- ✅ Sample patient data loader

### **Documentation**
- ✅ Complete API documentation
- ✅ Deployment guides
- ✅ Code comments throughout
- ✅ Troubleshooting guides

---

## 📂 Project Structure

```
heart-disease-ai/
│
├── 📄 Core Files
│   ├── heart_disease_prediction_complete.py  # Main training script
│   ├── api_integrated.py                     # FastAPI backend
│   ├── index.html                            # Web interface
│   ├── style_advanced.css                    # Styling
│   └── script_advanced.js                    # Frontend logic
│
├── 🚀 Quick Start Scripts
│   ├── start_api.bat                         # Windows launcher
│   └── start_api.sh                          # Mac/Linux launcher
│
├── 🤖 Model Files (Generated after training)
│   ├── xgboost_model.pkl                     # Trained XGBoost model
│   ├── xgboost_scaler.pkl                    # Feature scaler
│   └── xgboost_features.json                 # Feature configuration
│
├── 📊 Outputs (22+ visualization files)
│   ├── 01_target_distribution.png
│   ├── 02_ml_models_comparison.png
│   ├── 03_dl_models_comparison.png
│   └── ... (19+ more files)
│
├── 📚 Documentation
│   ├── README.md                             # This file
│   ├── DEPLOYMENT_GUIDE_COMPLETE.md          # Detailed deployment guide
│   ├── QUICK_START.md                        # Quick start guide
│   └── requirements.txt                      # Python dependencies
│
└── 📁 Data
    └── heart_disease.csv                     # Training dataset
```

---

## 🎓 Usage Instructions

### **For Training (First Time)**

1. **Prepare your data**
   - Ensure `heart_disease.csv` is in the project folder
   - Data should have a `target_final` column (0 = No Disease, 1 = Disease)

2. **Run training script**
   ```bash
   python heart_disease_prediction_complete.py
   ```

3. **Wait 15-30 minutes**
   - Script trains 20+ models
   - Generates 22+ visualizations
   - Saves best model automatically

4. **Check outputs folder**
   - Model files: `xgboost_model.pkl`, `xgboost_scaler.pkl`, `xgboost_features.json`
   - Visualizations: 22+ PNG files
   - CSV report: `21_complete_model_comparison.csv`

### **For Deployment (Every Time)**

1. **Start the API**
   ```bash
   # Windows
   start_api.bat
   
   # Mac/Linux
   ./start_api.sh
   ```

2. **Open web interface**
   - Double-click `index.html`
   - Or use HTTP server: `python -m http.server 8000`

3. **Make predictions**
   - Enter patient data manually
   - Or click "Load Sample Patient"
   - Click "Analyze Risk"

4. **View explanations**
   - Click "View Explanations"
   - Explore SHAP, LIME, and other visualizations

---

## 🔌 API Endpoints

### **Prediction Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single patient prediction |
| `/predict/batch` | POST | Multiple patients at once |

### **Explainability Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/explain/waterfall` | POST | SHAP waterfall plot (individual) |
| `/explain/force` | POST | SHAP force plot (individual) |
| `/explain/lime` | POST | LIME explanation (individual) |
| `/explain/summary` | GET | Global SHAP summary |
| `/explain/importance` | GET | Feature importance ranking |

### **Information Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/features/list` | GET | Feature list and encoding |
| `/docs` | GET | Interactive API documentation |

---

## 📊 Model Performance

Expected metrics (from training on heart disease dataset):

| Metric | Score | Grade |
|--------|-------|-------|
| Accuracy | 86.5% | Excellent ✅ |
| Precision | 86.9% | Excellent ✅ |
| Recall | 88.4% | Excellent ✅ |
| F1-Score | 87.6% | Excellent ✅ |
| ROC-AUC | ~0.93 | Excellent ✅ |

**Performance Grading:**
- **Excellent** (≥85%): Production-ready for research/educational use
- **Good** (70-85%): Acceptable for most scenarios
- **Fair** (60-70%): Needs improvement

---

## 🧪 Testing

### **Test API with Python**

```python
import requests

# Endpoint
url = "http://localhost:8001/predict"

# Patient data
patient = {
    "age": 65,
    "sex": "Male",
    "oldpeak": 2.5,
    "chest_pain": "Asymptomatic",
    "restingbp_final": 160,
    "chol_final": 300,
    "maxhr_final": 130,
    "fasting_bs": "Yes",
    "resting_ecg": "ST-T Abnormality",
    "exercise_angina": "Yes",
    "st_slope": "Flat"
}

# Make prediction
response = requests.post(url, json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### **Test API with cURL**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "sex": "Male",
    "oldpeak": 2.5,
    "chest_pain": "Asymptomatic",
    "restingbp_final": 160,
    "chol_final": 300,
    "maxhr_final": 130,
    "fasting_bs": "Yes",
    "resting_ecg": "ST-T Abnormality",
    "exercise_angina": "Yes",
    "st_slope": "Flat"
  }'
```

---

## 🐛 Troubleshooting

### **Problem: API won't start**
```
Error: ModuleNotFoundError: No module named 'fastapi'
```
**Solution:**
```bash
pip install fastapi uvicorn[standard]
```

### **Problem: Model file not found**
```
Error: FileNotFoundError: xgboost_model.pkl
```
**Solution:**
1. Run training script: `python heart_disease_prediction_complete.py`
2. Wait for it to complete (15-30 minutes)
3. Check `outputs/` folder for model files

### **Problem: Port already in use**
```
Error: [Errno 48] Address already in use
```
**Solution:**
```bash
# Change port in api_integrated.py (line ~700)
uvicorn.run(..., port=8002)  # Use different port

# Or kill process using port 8001
# Windows: netstat -ano | findstr :8001
# Mac/Linux: lsof -i :8001
```

### **Problem: SHAP plots fail**
```
Error: ImportError: No module named 'shap'
```
**Solution:**
```bash
pip install shap
```

---

## 🎤 Presentation Tips

### **Demo Flow (5-10 minutes)**

1. **Introduction (1 min)**
   - "This is an end-to-end heart disease prediction system with Explainable AI"
   - Show project structure

2. **Training Pipeline (2 min)**
   - Run training script (or show pre-recorded)
   - Highlight: 20+ models trained, best selected automatically
   - Show generated visualizations

3. **Web Interface (3 min)**
   - Load sample patient
   - Show prediction with risk level
   - Demonstrate probability gauge
   - Show risk/protective factors

4. **Explainability (3 min)**
   - Navigate to Explanations tab
   - Show SHAP waterfall plot
   - Explain force plot
   - Show LIME local explanation
   - Display global feature importance

5. **API Demo (1 min)**
   - Show `/docs` endpoint
   - Demonstrate interactive API testing

### **Key Talking Points**

✅ **Comprehensive**: 20+ models evaluated (ML + DL + Hybrid)  
✅ **Explainable**: SHAP, LIME, multiple visualization techniques  
✅ **Production-Ready**: Complete API with validation, error handling  
✅ **User-Friendly**: Modern web interface with real-time predictions  
✅ **Well-Documented**: Extensive guides and inline comments  

---

## ⚠️ **Important Disclaimer**

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This system is:
- ✅ Suitable for educational demonstrations
- ✅ Suitable for research projects
- ✅ Suitable for learning ML/AI concepts
- ❌ **NOT** FDA approved
- ❌ **NOT** for clinical diagnosis
- ❌ **NOT** a replacement for medical professionals

**Always consult qualified healthcare professionals for medical decisions.**

---

## 📄 License

This project is for educational purposes as part of a Senior Design Project.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository (Heart Disease Dataset)
- Scikit-learn, TensorFlow, XGBoost developers
- SHAP and LIME library authors
- Open-source ML/AI community

---

## 📞 Support

For issues or questions:
1. Check the `DEPLOYMENT_GUIDE_COMPLETE.md` for detailed instructions
2. Review the troubleshooting section above
3. Check API documentation at `http://localhost:8001/docs`

---

## ✅ Final Checklist

Before your presentation:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Training script ran successfully
- [ ] Model files exist in `outputs/` folder
- [ ] All 22+ visualizations generated
- [ ] API starts without errors (`start_api.bat` or `./start_api.sh`)
- [ ] Web interface loads (`index.html` opens)
- [ ] Sample prediction works
- [ ] SHAP plots display correctly
- [ ] LIME explanations work
- [ ] Documentation reviewed
- [ ] Code is clean and well-commented

---

## 🚀 Ready to Deploy!

Your complete Heart Disease Prediction system includes:

✅ **Training Pipeline** - Automatic model selection  
✅ **Production API** - FastAPI with full documentation  
✅ **Web Interface** - Interactive and user-friendly  
✅ **Explainable AI** - SHAP, LIME, visualizations  
✅ **Documentation** - Comprehensive guides  

**Start now:**
```bash
# 1. Start API
start_api.bat  # (Windows) or ./start_api.sh (Mac/Linux)

# 2. Open browser
# Double-click index.html

# 3. Make prediction!
```

---

**Good luck with your Senior Design Project!** 🎓🫀

**Version:** 3.0.0  
**Last Updated:** March 24, 2026  
**Status:** Production Ready ✅
