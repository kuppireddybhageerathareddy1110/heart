# Heart Disease Prediction Using Explainable AI

## 🎓 Senior Design Project

A comprehensive machine learning and deep learning system for predicting heart disease with full explainability using SHAP, LIME, and other interpretability techniques.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset Requirements](#dataset-requirements)
6. [Usage Guide](#usage-guide)
7. [Script Sections](#script-sections)
8. [Output Files](#output-files)
9. [Model Performance](#model-performance)
10. [Explainability Techniques](#explainability-techniques)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a complete pipeline for heart disease prediction using:

- **11 Traditional ML Models**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, AdaBoost, Extra Trees, SVM, KNN, Naive Bayes
- **7 Deep Learning Architectures**: ANN, Deep ANN, Dropout Net, Batch Normalization Net, Combined Net, Wide-Deep Net, Residual Net
- **4 Hybrid Ensemble Strategies**: Simple Average, Weighted Average, Stacking, Voting
- **Complete Explainability**: SHAP, LIME, Partial Dependence Plots, Feature Interactions

### Key Highlights

✅ **Comprehensive**: 20+ models covering all major ML/DL architectures  
✅ **Explainable**: Full SHAP, LIME, and PDP analysis  
✅ **Production-Ready**: Model saving, deployment guide included  
✅ **Well-Documented**: Clear sections with extensive comments  
✅ **Visualizations**: 22+ professional plots and charts  

---

## ✨ Features

### Machine Learning Models
- Traditional algorithms (Logistic Regression, SVM, Trees)
- Ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Advanced boosting (AdaBoost, Extra Trees)

### Deep Learning Models
- Basic Neural Networks (ANN)
- Deep architectures (4-5 hidden layers)
- Regularization techniques (Dropout, Batch Normalization)
- Advanced architectures (Wide-Deep, Residual Networks)

### Hybrid Ensemble Models
- Simple and weighted averaging
- Stacking with meta-learner
- Voting ensembles

### Explainability Tools
- **SHAP Analysis**: Global and local interpretability
- **LIME**: Local explanations for individual predictions
- **Partial Dependence Plots**: Feature effect visualization
- **Feature Interactions**: SHAP interaction values
- **Correlation Analysis**: Feature relationship heatmaps

### Evaluation Metrics
- Confusion Matrix
- ROC Curve & AUC
- Precision-Recall Curve
- Calibration Curve
- Decision Curve Analysis
- Complete Classification Report

---

## 📦 Requirements

### Python Version
- Python 3.8 or higher

### Core Libraries
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Machine Learning
```
xgboost>=1.5.0
lightgbm>=3.3.0
```

### Deep Learning
```
tensorflow>=2.8.0
keras>=2.8.0
```

### Explainability
```
shap>=0.40.0
lime>=0.2.0
```

### Installation Command
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm tensorflow shap lime
```

---

## 🚀 Installation

### Step 1: Clone or Download
Download the `heart_disease_prediction_complete.py` file.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm tensorflow
pip install shap lime
```

### Step 3: Verify Installation
```python
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import shap
import lime

print("All libraries installed successfully!")
```

---

## 📊 Dataset Requirements

### Expected Format
- CSV file with features and target column
- Target column: Binary (0 = No Disease, 1 = Disease)
- No missing values (or handle them before loading)

### Recommended Features
Your dataset should include clinical features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression (oldpeak)
- Slope of peak exercise ST segment
- Number of major vessels
- Thalassemia

### Example Data Structure
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
...
```

### Data Preprocessing
The script includes:
- Automatic train-test splitting (80/20)
- Standard scaling (mean=0, std=1)
- Feature engineering (optional - customize for your data)

---

## 💻 Usage Guide

### Quick Start

1. **Prepare Your Data**
   ```python
   # Ensure your CSV is in the correct format
   # Column names should match your dataset
   ```

2. **Update File Path**
   ```python
   # In the script, locate line in main_pipeline():
   data_path = 'heart_disease.csv'  # Change to your file path
   ```

3. **Customize Feature Engineering** (if needed)
   ```python
   # In preprocess_data() function, uncomment and modify:
   if 'age' in X.columns and 'chol' in X.columns:
       X['age_chol_interaction'] = X['age'] * X['chol']
   ```

4. **Run the Pipeline**
   ```python
   # Uncomment in main_pipeline() function
   # Then execute:
   python heart_disease_prediction_complete.py
   ```

### Step-by-Step Execution

You can also run sections individually:

```python
# 1. Load data
df = load_and_explore_data('heart_disease.csv')

# 2. Preprocess
X, y, feature_names = preprocess_data(df)

# 3. Split and scale
X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)

# 4. Train ML models
ml_models, ml_results = train_ml_models(X_train, X_test, y_train, y_test)

# 5. Train DL models
dl_models, dl_results, histories = train_dl_models(X_train, X_test, y_train, y_test)

# 6. Create ensemble
ensemble_results, predictions, meta_model = create_hybrid_ensemble(
    X_train, X_test, y_train, y_test, ml_models, dl_models
)

# 7. Explainability analysis
perform_shap_analysis(ml_models['XGBoost'], X_train, X_test, feature_names, 'XGBoost', 'tree')
```

---

## 📁 Script Sections

### Section 1: Library Imports
- All necessary imports organized by category
- Random seed setting for reproducibility

### Section 2: Data Loading & Exploration
- `load_and_explore_data()`: Load CSV and show statistics
- Missing value detection
- Target distribution analysis

### Section 3: Data Preprocessing
- `preprocess_data()`: Feature engineering
- Create interaction features
- Handle categorical variables

### Section 4: Train-Test Split
- `prepare_train_test_split()`: 80/20 split
- StandardScaler application
- Stratified sampling

### Section 5: Traditional ML Models
- `train_ml_models()`: Train 11 ML algorithms
- Performance comparison
- Visualization

### Section 6: Deep Learning Models
- `train_dl_models()`: Train 7 DL architectures
- Early stopping & learning rate reduction
- Training curve visualization

### Section 7: Hybrid Ensemble
- `create_hybrid_ensemble()`: 4 ensemble strategies
- Meta-learner training
- Performance comparison

### Section 8: Model Evaluation
- Confusion matrices
- ROC & PR curves
- Calibration curves
- Decision curve analysis

### Section 9: SHAP Analysis
- `perform_shap_analysis()`: Complete SHAP workflow
- Summary plots (beeswarm)
- Feature importance
- Dependence plots
- Waterfall plots
- Force plots

### Section 10: LIME Analysis
- `perform_lime_analysis()`: Local explanations
- Individual prediction interpretation

### Section 11: Partial Dependence Plots
- `create_partial_dependence_plots()`: Feature effects
- Non-linear relationship visualization

### Section 12: Correlation Analysis
- `analyze_feature_correlations()`: Feature relationships
- Correlation heatmap

### Section 13: Comprehensive Comparison
- `create_comprehensive_comparison()`: All models
- Category-wise analysis
- Performance scatter plots

### Section 14: Model Deployment
- `save_best_model()`: Export for production
- Deployment guide generation

### Section 15: Main Pipeline
- `main_pipeline()`: Complete execution
- Orchestrates all sections

---

## 📊 Output Files

After running the script, you'll get 22+ output files:

### Visualizations (PNG)
1. `01_target_distribution.png` - Class distribution
2. `02_ml_models_comparison.png` - ML model performance
3. `03_dl_models_comparison.png` - DL model performance
4. `04_training_curves.png` - Loss and accuracy curves
5. `05_hybrid_ensemble_comparison.png` - Ensemble strategies
6. `06_confusion_matrix_*.png` - Confusion matrices
7. `07_roc_curve_*.png` - ROC curves
8. `08_pr_curve_*.png` - Precision-Recall curves
9. `09_calibration_curve_*.png` - Calibration plots
10. `10_decision_curve_*.png` - Decision curve analysis
11. `11_shap_summary_*.png` - SHAP summary (beeswarm)
12. `12_shap_importance_*.png` - SHAP feature importance
13. `13_shap_dependence_*.png` - SHAP dependence plots
14. `14_shap_waterfall_*.png` - SHAP waterfall plots
15. `15_shap_force_*.png` - SHAP force plots
16. `16_shap_interaction_heatmap_*.png` - Feature interactions
17. `17_shap_interaction_dependence_*.png` - Interaction plots
18. `18_lime_explanation_*.png` - LIME explanations
19. `19_partial_dependence_plots.png` - PDP plots
20. `20_correlation_matrix.png` - Feature correlations
21. `21_complete_model_comparison.csv` - Results table
22. `22_comprehensive_comparison.png` - All models comparison

### Model Files
- `best_model_model.pkl` - Trained model
- `best_model_scaler.pkl` - Feature scaler
- `best_model_features.json` - Feature names
- `deployment_guide.txt` - Deployment instructions

---

## 📈 Model Performance

### Expected Performance Ranges

**Traditional ML Models**
- Accuracy: 75-88%
- Best performers: XGBoost, Random Forest, Gradient Boosting

**Deep Learning Models**
- Accuracy: 78-86%
- Best performers: BatchNorm Net, Combined Net

**Hybrid Ensemble**
- Accuracy: 85-92%
- Best performer: Stacked Hybrid (typically highest)

### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

---

## 🔍 Explainability Techniques

### 1. SHAP (SHapley Additive exPlanations)

**Global Interpretability**
- Summary plots show overall feature importance
- Bar plots rank features by impact

**Local Interpretability**
- Waterfall plots explain individual predictions
- Force plots show feature contributions

**Feature Interactions**
- Interaction heatmaps reveal feature dependencies
- Interaction dependence plots show combined effects

### 2. LIME (Local Interpretable Model-agnostic Explanations)

- Explains individual predictions
- Works with any black-box model
- Shows top contributing features

### 3. Partial Dependence Plots (PDP)

- Visualize feature effects on predictions
- Show marginal effect of features
- Reveal non-linear relationships

### 4. Feature Correlation Analysis

- Correlation heatmaps
- Identifies multicollinearity
- Helps feature selection

---

## 🚀 Deployment

### Saving the Best Model

The script automatically saves:
```python
best_model_model.pkl      # Trained model
best_model_scaler.pkl     # Feature scaler
best_model_features.json  # Feature configuration
deployment_guide.txt      # Usage instructions
```

### Loading for Prediction

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('best_model_model.pkl')
scaler = joblib.load('best_model_scaler.pkl')

# Prepare new patient data
new_patient = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Scale features
new_patient_scaled = scaler.transform(new_patient)

# Make prediction
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print(f'Prediction: {prediction[0]}')
print(f'Disease Probability: {probability[0][1]:.2%}')
```

### Flask API Example

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('best_model_model.pkl')
scaler = joblib.load('best_model_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```
Problem: ModuleNotFoundError: No module named 'shap'
Solution: pip install shap
```

**2. Memory Issues**
```
Problem: MemoryError during model training
Solution: Reduce batch size or use fewer models
```

**3. SHAP GPU Issues**
```
Problem: SHAP fails with TensorFlow GPU
Solution: Use CPU for SHAP analysis or use smaller background samples
```

**4. Feature Mismatch**
```
Problem: Feature count doesn't match
Solution: Ensure feature engineering is consistent
```

**5. Convergence Warnings**
```
Problem: Logistic Regression doesn't converge
Solution: Increase max_iter parameter
```

### Performance Optimization

**For Large Datasets (>100k samples)**
```python
# Use sampling for SHAP
background = shap.sample(X_train, 100)  # Instead of using all data

# Reduce epochs for DL
epochs = 50  # Instead of 100

# Use fewer models
# Comment out slower models like SVM
```

**For Small Datasets (<1k samples)**
```python
# Increase cross-validation folds
cv = 10  # Instead of 5

# Use simpler models
# Focus on traditional ML instead of complex DL
```

---

## 📝 Customization Guide

### Adding New Models

```python
# In Section 5 (ML Models)
def train_ml_models(X_train, X_test, y_train, y_test):
    ml_models = {
        # ... existing models ...
        'Your Custom Model': YourModelClass(params),
    }
```

### Modifying Feature Engineering

```python
# In Section 3
def preprocess_data(df, target_column='target'):
    # Add your custom features
    X['custom_feature'] = X['feature1'] * X['feature2']
    X['log_feature'] = np.log(X['feature3'] + 1)
```

### Changing Hyperparameters

```python
# XGBoost example
xgb_model = XGBClassifier(
    n_estimators=300,        # Modify this
    max_depth=6,             # Modify this
    learning_rate=0.05,      # Modify this
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## 📚 References

### Papers
1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining predictions.

### Documentation
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [LIME](https://lime-ml.readthedocs.io/)

---

## 👨‍💻 Author

**Senior Design Project**  
Heart Disease Prediction Using Explainable AI

---

## 📄 License

This project is for educational purposes as part of a senior design project.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for heart disease datasets
- Open-source ML/DL community
- SHAP and LIME developers

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review function docstrings
3. Verify data format matches requirements

---

**Good luck with your Senior Design Project! 🎓**
