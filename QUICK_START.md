# QUICK START GUIDE
# Heart Disease Prediction - Senior Design Project

## 📋 IMMEDIATE STEPS TO RUN

### Step 1: Install Dependencies (2 minutes)
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm tensorflow shap lime
```

### Step 2: Prepare Your Data
- Ensure your CSV has a target column (0 = No Disease, 1 = Disease)
- Save as 'heart_disease.csv' in the same folder as the script

### Step 3: Edit the Script (1 minute)
Open `heart_disease_prediction_complete.py` and:

1. Go to line ~1850 (in main_pipeline function)
2. Find this line:
   ```python
   # df = load_and_explore_data('heart_disease.csv')
   ```
3. Remove ALL the `#` symbols and triple quotes around the pipeline code
4. Change 'heart_disease.csv' to your actual file path

### Step 4: Run the Script
```bash
python heart_disease_prediction_complete.py
```

## 🎯 WHAT YOU'LL GET

### Console Output
- Progress updates for each section
- Performance tables for all models
- Best model identification

### 22+ Visualization Files
1. Target distribution
2. ML models comparison
3. DL models comparison
4. Training curves
5. Ensemble comparison
6. Confusion matrices
7. ROC curves
8. Precision-Recall curves
9. Calibration curves
10. Decision curves
11. SHAP summary plots
12. SHAP feature importance
13. SHAP dependence plots
14. SHAP waterfall plots
15. SHAP force plots
16. Feature interaction heatmaps
17. Interaction dependence plots
18. LIME explanations
19. Partial dependence plots
20. Correlation matrix
21. Comprehensive comparison
22. Performance scatter plots

### Model Files for Deployment
- best_model_model.pkl
- best_model_scaler.pkl
- best_model_features.json
- deployment_guide.txt

## ⏱️ EXPECTED RUNTIME

- Small dataset (<1000 samples): 5-10 minutes
- Medium dataset (1000-10000 samples): 15-30 minutes
- Large dataset (>10000 samples): 30-60 minutes

## 🔧 COMMON ADJUSTMENTS

### If You Have Different Column Names

Find this section in the script (around line 150):
```python
def preprocess_data(df, target_column='target'):
```

Change 'target' to your actual target column name.

### If You Want Faster Execution

1. Reduce number of models:
   - Comment out slow models (SVM, some DL models)
   
2. Reduce epochs for Deep Learning:
   ```python
   epochs=50  # Instead of 100
   ```

3. Use smaller SHAP samples:
   ```python
   background = shap.sample(X_train, 50)  # Instead of 100
   ```

### If You Want More Detail

1. Change `verbose=0` to `verbose=1` in model training
2. Increase number of samples for LIME analysis
3. Add more features to Partial Dependence Plots

## 📊 INTERPRETING RESULTS

### Model Comparison Table
- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted positives, how many are correct (higher is better)
- **Recall**: Of actual positives, how many we caught (higher is better)
- **F1-Score**: Balance between precision and recall (higher is better)
- **ROC-AUC**: Overall model performance (closer to 1 is better)

### SHAP Plots
- **Red dots**: Higher feature values
- **Blue dots**: Lower feature values
- **Position on x-axis**: Impact on prediction (right = increases risk, left = decreases risk)

### Confusion Matrix
```
                Predicted
              No    Yes
Actual  No   [TN]  [FP]   ← False Positives (False Alarm)
        Yes  [FN]  [TP]   ← True Positives (Correct Detection)
              ↑
         False Negatives
         (Missed Cases)
```

## 🚨 TROUBLESHOOTING

### Error: "No module named 'xyz'"
```bash
pip install xyz
```

### Error: "Memory Error"
- Reduce number of models
- Use smaller dataset sample
- Close other applications

### Error: "Feature count mismatch"
- Check that all rows have same number of features
- Verify no NaN values in data

### Warning: "Model did not converge"
- This is usually OK
- Can increase max_iter if concerned

## 📞 QUICK HELP

### Check Your Data Format
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
```

### Test One Model First
Instead of running full pipeline, test with one model:
```python
# Load and prepare data
df = load_and_explore_data('heart_disease.csv')
X, y, feature_names = preprocess_data(df)
X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)

# Train just XGBoost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## ✅ CHECKLIST BEFORE RUNNING

- [ ] All libraries installed
- [ ] Data file ready and accessible
- [ ] Target column identified
- [ ] Script edited (uncommented main_pipeline)
- [ ] File path updated in script
- [ ] Output directory exists

## 🎓 FOR YOUR PRESENTATION

### Key Points to Highlight

1. **Comprehensiveness**: 20+ models tested
2. **Best Performance**: Show top 3 models from comparison table
3. **Explainability**: SHAP shows which features matter most
4. **Clinical Relevance**: LIME explains individual patient predictions
5. **Production Ready**: Saved model can be deployed

### Recommended Visualizations to Present

1. Complete model comparison (shows all your work)
2. SHAP summary plot (feature importance)
3. Confusion matrix of best model (performance)
4. ROC curve (clinical decision making)
5. SHAP waterfall for one patient (interpretability)

### Talking Points

- "Tested 20+ different models to find the best approach"
- "Achieved X% accuracy with explainable predictions"
- "Every prediction can be explained using SHAP values"
- "Ready for clinical deployment with saved model files"
- "Follows best practices: train/test split, cross-validation, multiple metrics"

## 🎉 YOU'RE READY!

Run the script and watch your comprehensive heart disease prediction system come to life!

```bash
python heart_disease_prediction_complete.py
```

Good luck with your Senior Design Project! 🚀
