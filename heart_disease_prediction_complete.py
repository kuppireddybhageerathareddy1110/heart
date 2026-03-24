"""
================================================================================
HEART DISEASE PREDICTION USING EXPLAINABLE AI
Senior Design Project
Author: [Your Name]
Date: March 2026

This script implements a comprehensive heart disease prediction system using:
- Traditional Machine Learning models
- Deep Learning architectures
- Hybrid ensemble models
- Explainable AI techniques (SHAP, LIME)
- Model interpretability and visualization
================================================================================
"""

# ================================================================================
# SECTION 1: IMPORT LIBRARIES
# ================================================================================

import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs("outputs", exist_ok=True)
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost and LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Deep Learning - TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Add,
    Conv1D, Flatten, LSTM, GRU
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Explainable AI
import shap
import lime
import lime.lime_tabular

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("HEART DISEASE PREDICTION - EXPLAINABLE AI PROJECT")
print("=" * 80)
print("\n✓ All libraries imported successfully\n")


# ================================================================================
# SECTION 2: DATA LOADING AND EXPLORATION
# ================================================================================

print("=" * 80)
print("SECTION 2: DATA LOADING AND EXPLORATION")
print("=" * 80)

def load_and_explore_data(filepath):
    """
    Load and perform initial exploration of the dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing heart disease data
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    print("\n📂 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Samples: {df.shape[0]}")
    print(f"  Features: {df.shape[1]}")
    
    print("\n📊 Dataset Overview:")
    print(df.head())
    
    print("\n📋 Dataset Information:")
    print(df.info())
    
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values found")
    else:
        print(missing[missing > 0])
    
    print("\n🎯 Target Distribution:")
    target_counts = df.iloc[:, -1].value_counts()
    print(target_counts)
    print(f"  Class balance: {target_counts.min() / target_counts.max():.2%}")
    
    return df

# Note: Replace 'heart_disease.csv' with your actual data file path

df = load_and_explore_data('heart_disease.csv')

TARGET = "target_final"

# ================================================================================
# SECTION 3: DATA PREPROCESSING AND FEATURE ENGINEERING
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 3: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("=" * 80)

def preprocess_data(df, target_column='target_final'):
    

    """
    Preprocess data and create engineered features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
    target_column : str
        Name of the target column
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    feature_names : list
        List of feature names
    """
    print("\n🔧 Starting data preprocessing...")
    
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    print(f"  ✓ Features extracted: {len(feature_names)}")
    print(f"  ✓ Target extracted: {y.name if hasattr(y, 'name') else 'target'}")
    
    # Feature Engineering (example features - adjust based on your data)
    print("\n🛠️ Engineering new features...")
    
    # Example: Create interaction features (adjust column names as needed)
    # Uncomment and modify based on your actual column names
    """
    if 'age' in X.columns and 'chol' in X.columns:
        X['age_chol_interaction'] = X['age'] * X['chol']
    
    if 'trestbps' in X.columns and 'chol' in X.columns:
        X['bp_chol_ratio'] = X['trestbps'] / (X['chol'] + 1)
    
    if 'thalach' in X.columns and 'age' in X.columns:
        X['max_hr_age_ratio'] = X['thalach'] / (X['age'] + 1)
        X['cardiac_load'] = X['thalach'] * X['age']
    
    if 'oldpeak' in X.columns and 'thalach' in X.columns:
        X['stress_index'] = X['oldpeak'] * (220 - X['thalach'])
    """
    
    print(f"  ✓ Total features after engineering: {X.shape[1]}")
    
    return X, y, X.columns.tolist()


def visualize_data_distribution(df, target_column='target'):
    """
    Visualize data distributions and relationships
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to visualize
    target_column : str
        Name of the target column
    """
    print("\n📊 Creating data visualizations...")
    
    # Target distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    df[target_column].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.title('Target Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    df[target_column].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                          colors=['#3498db', '#e74c3c'])
    plt.title('Target Proportion', fontsize=14, fontweight='bold')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('outputs/01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Target distribution plot saved")


# ================================================================================
# SECTION 4: TRAIN-TEST SPLIT AND SCALING
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 4: TRAIN-TEST SPLIT AND SCALING")
print("=" * 80)

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data and apply standardization
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Split and scaled datasets
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\n🔪 Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  ✓ Training set: {X_train.shape[0]} samples")
    print(f"  ✓ Test set: {X_test.shape[0]} samples")
    print(f"  ✓ Split ratio: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
    
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  ✓ StandardScaler applied")
    print(f"  ✓ Feature means: {X_train_scaled.mean():.2e}")
    print(f"  ✓ Feature stds: {X_train_scaled.std():.2e}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ================================================================================
# SECTION 5: TRADITIONAL MACHINE LEARNING MODELS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 5: TRADITIONAL MACHINE LEARNING MODELS")
print("=" * 80)

def train_ml_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple ML models
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    results : pandas.DataFrame
        Model performance metrics
    """
    print("\n🤖 Training Traditional ML Models...")
    print("-" * 80)
    
    # Define models
    ml_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    trained_models = {}
    
    for name, model in ml_models.items():
        print(f"\n  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        if y_pred_proba is not None:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            auc_score = np.nan
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_score
        })
        
        trained_models[name] = model
        
        print(f"    ✓ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("ML MODELS PERFORMANCE SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return trained_models, results_df


def visualize_ml_results(results_df):
    """
    Visualize ML model comparison
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Model performance metrics
    """
    print("\n📊 Creating ML model comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    results_sorted = results_df.sort_values('Accuracy')
    axes[0, 0].barh(results_sorted['Model'], results_sorted['Accuracy'], color='#3498db')
    axes[0, 0].set_xlabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim([0, 1])
    
    # All metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    top_5_models = results_df.head(5)
    x = np.arange(len(top_5_models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0, 1].bar(x + i*width, top_5_models[metric], width, label=metric)
    
    axes[0, 1].set_xlabel('Models', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Top 5 Models - All Metrics', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x + width * 1.5)
    axes[0, 1].set_xticklabels(top_5_models['Model'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # ROC-AUC comparison
    roc_data = results_df.dropna(subset=['ROC-AUC']).sort_values('ROC-AUC')
    axes[1, 0].barh(roc_data['Model'], roc_data['ROC-AUC'], color='#e74c3c')
    axes[1, 0].set_xlabel('ROC-AUC Score', fontsize=12)
    axes[1, 0].set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    
    # F1-Score comparison
    f1_sorted = results_df.sort_values('F1-Score')
    axes[1, 1].barh(f1_sorted['Model'], f1_sorted['F1-Score'], color='#2ecc71')
    axes[1, 1].set_xlabel('F1-Score', fontsize=12)
    axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('outputs/02_ml_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ ML comparison plots saved")


# ================================================================================
# SECTION 6: DEEP LEARNING MODELS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 6: DEEP LEARNING MODELS")
print("=" * 80)

# Define Early Stopping and Learning Rate Reduction
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)


def create_ann_model(input_dim):
    """Standard Artificial Neural Network"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='ANN')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_deep_ann_model(input_dim):
    """Deep Artificial Neural Network"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='DeepANN')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_dropout_model(input_dim):
    """Neural Network with Dropout"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ], name='DropoutNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_batchnorm_model(input_dim):
    """Neural Network with Batch Normalization"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ], name='BatchNormNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_combined_model(input_dim):
    """Neural Network with Dropout + Batch Normalization"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ], name='CombinedNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_wide_deep_model(input_dim):
    """Wide and Deep Model Architecture"""
    input_layer = Input(shape=(input_dim,), name='input')
    
    # Wide path
    wide = Dense(64, activation='relu', name='wide_1')(input_layer)
    
    # Deep path
    deep = Dense(128, activation='relu', name='deep_1')(input_layer)
    deep = Dense(64, activation='relu', name='deep_2')(deep)
    deep = Dense(32, activation='relu', name='deep_3')(deep)
    
    # Merge
    merged = tf.keras.layers.concatenate([wide, deep])
    merged = Dense(64, activation='relu', name='merged')(merged)
    output = Dense(1, activation='sigmoid', name='output')(merged)
    
    model = Model(inputs=input_layer, outputs=output, name='WideDeepNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_residual_model(input_dim):
    """Residual Neural Network"""
    input_layer = Input(shape=(input_dim,), name='input')
    
    x = Dense(64, activation='relu', name='layer_1')(input_layer)
    
    # Residual block 1
    res = Dense(64, activation='relu', name='res_1')(x)
    res = Dense(64, activation='relu', name='res_2')(res)
    x = Add(name='add_1')([x, res])
    
    # Residual block 2
    res2 = Dense(64, activation='relu', name='res_3')(x)
    res2 = Dense(64, activation='relu', name='res_4')(res2)
    x = Add(name='add_2')([x, res2])
    
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=input_layer, outputs=output, name='ResNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_dl_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple DL models
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    results : pandas.DataFrame
        Model performance metrics
    histories : dict
        Training histories for each model
    """
    print("\n🧠 Training Deep Learning Models...")
    print("-" * 80)
    
    input_dim = X_train.shape[1]
    
    # Define DL models
    dl_model_creators = {
        'ANN': create_ann_model,
        'Deep ANN': create_deep_ann_model,
        'Dropout Net': create_dropout_model,
        'BatchNorm Net': create_batchnorm_model,
        'Combined Net': create_combined_model,
        'Wide-Deep Net': create_wide_deep_model,
        'Residual Net': create_residual_model
    }
    
    results = []
    trained_models = {}
    histories = {}
    
    for name, creator_func in dl_model_creators.items():
        print(f"\n  Training {name}...")
        
        # Create model
        model = creator_func(input_dim)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_score
        })
        
        trained_models[name] = model
        histories[name] = history
        
        print(f"    ✓ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("DL MODELS PERFORMANCE SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return trained_models, results_df, histories


def visualize_dl_results(results_df, histories):
    """
    Visualize DL model comparison and training curves
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Model performance metrics
    histories : dict
        Training histories
    """
    print("\n📊 Creating DL model comparison visualizations...")
    
    # Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    results_sorted = results_df.sort_values('Accuracy')
    axes[0].barh(results_sorted['Model'], results_sorted['Accuracy'], color='#9b59b6')
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('DL Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0, 1])
    
    # All metrics for top models
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    top_3_models = results_df.head(3)
    x = np.arange(len(top_3_models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        axes[1].bar(x + i*width, top_3_models[metric], width, label=metric)
    
    axes[1].set_xlabel('Models', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Top 3 DL Models - All Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels(top_3_models['Model'], rotation=15, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('outputs/03_dl_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training curves for best model
    best_model_name = results_df.iloc[0]['Model']
    best_history = histories[best_model_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss curves
    axes[0].plot(best_history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(best_history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training vs Validation Loss - {best_model_name}', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(best_history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(best_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Training vs Validation Accuracy - {best_model_name}', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/04_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ DL comparison and training curves saved")


# ================================================================================
# SECTION 7: HYBRID ENSEMBLE MODELS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 7: HYBRID ENSEMBLE MODELS")
print("=" * 80)

def create_hybrid_ensemble(X_train, X_test, y_train, y_test, ml_models, dl_models):
    """
    Create hybrid ensemble combining ML and DL models
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    ml_models : dict
        Trained ML models
    dl_models : dict
        Trained DL models
        
    Returns:
    --------
    results : pandas.DataFrame
        Ensemble performance metrics
    predictions : dict
        Predictions from different ensemble strategies
    """
    print("\n🔄 Creating Hybrid Ensemble Models...")
    print("-" * 80)
    
    # Get best ML model (XGBoost)
    xgb_model = ml_models['XGBoost']
    
    # Get best DL model
    best_dl_name = 'BatchNorm Net'  # or use the actual best from results
    dl_model = dl_models.get(best_dl_name, list(dl_models.values())[0])
    
    # Get predictions
    print("\n  Generating predictions from base models...")
    xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    dl_train_probs = dl_model.predict(X_train, verbose=0).flatten()
    dl_test_probs = dl_model.predict(X_test, verbose=0).flatten()
    
    # Strategy 1: Simple Averaging
    print("\n  Strategy 1: Simple Averaging...")
    avg_probs = (xgb_test_probs + dl_test_probs) / 2
    avg_preds = (avg_probs > 0.5).astype(int)
    
    # Strategy 2: Weighted Averaging (favor better model)
    print("  Strategy 2: Weighted Averaging...")
    weighted_probs = (0.6 * xgb_test_probs) + (0.4 * dl_test_probs)
    weighted_preds = (weighted_probs > 0.5).astype(int)
    
    # Strategy 3: Stacking with Meta-Learner
    print("  Strategy 3: Stacking with Meta-Learner...")
    meta_X_train = np.column_stack([xgb_train_probs, dl_train_probs])
    meta_X_test = np.column_stack([xgb_test_probs, dl_test_probs])
    
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X_train, y_train)
    stack_preds = meta_model.predict(meta_X_test)
    stack_probs = meta_model.predict_proba(meta_X_test)[:, 1]
    
    # Strategy 4: Voting Ensemble
    print("  Strategy 4: Voting Ensemble...")
    xgb_preds = (xgb_test_probs > 0.5).astype(int)
    dl_preds = (dl_test_probs > 0.5).astype(int)
    voting_preds = ((xgb_preds + dl_preds) > 0).astype(int)  # Majority vote
    
    # Evaluate all strategies
    results = []
    
    strategies = {
        'XGBoost Only': (xgb_test_probs, (xgb_test_probs > 0.5).astype(int)),
        'DL Only': (dl_test_probs, (dl_test_probs > 0.5).astype(int)),
        'Simple Average': (avg_probs, avg_preds),
        'Weighted Average': (weighted_probs, weighted_preds),
        'Stacked Hybrid': (stack_probs, stack_preds),
        'Voting Ensemble': (None, voting_preds)
    }
    
    for name, (probs, preds) in strategies.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='binary')
        rec = recall_score(y_test, preds, average='binary')
        f1 = f1_score(y_test, preds, average='binary')
        
        if probs is not None:
            auc_score = roc_auc_score(y_test, probs)
        else:
            auc_score = np.nan
        
        results.append({
            'Strategy': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_score
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("HYBRID ENSEMBLE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    predictions = {
        'xgb_probs': xgb_test_probs,
        'dl_probs': dl_test_probs,
        'stack_probs': stack_probs,
        'stack_preds': stack_preds
    }
    
    return results_df, predictions, meta_model


def visualize_ensemble_results(results_df):
    """
    Visualize ensemble comparison
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Ensemble performance metrics
    """
    print("\n📊 Creating ensemble comparison visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    results_sorted = results_df.sort_values('Accuracy')
    axes[0].barh(results_sorted['Strategy'], results_sorted['Accuracy'], 
                 color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Hybrid Ensemble Strategy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0, 1])
    axes[0].axvline(x=results_df['Accuracy'].mean(), color='red', 
                    linestyle='--', linewidth=2, label='Mean Accuracy')
    axes[0].legend()
    
    # All metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[1].bar(x + i*width, results_df[metric], width, label=metric)
    
    axes[1].set_xlabel('Strategy', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(results_df['Strategy'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('outputs/05_hybrid_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Ensemble comparison plots saved")


# ================================================================================
# SECTION 8: MODEL EVALUATION AND VISUALIZATION
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 8: MODEL EVALUATION AND VISUALIZATION")
print("=" * 80)

def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'outputs/06_confusion_matrix_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/07_roc_curve_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model'):
    """
    Plot Precision-Recall curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=3, label=model_name)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/08_pr_curve_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_curve(y_true, y_pred_proba, model_name='Model'):
    """
    Plot calibration curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Probability', fontsize=12)
    plt.title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/09_calibration_curve_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_decision_curve_analysis(y_true, y_pred_proba, model_name='Model'):
    """
    Plot decision curve analysis
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit = []
    
    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))
        
        nb = (TP / len(y_true)) - (FP / len(y_true)) * (t / (1 - t))
        net_benefit.append(nb)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit, linewidth=2, label=model_name)
    plt.axhline(y=0, color='gray', linestyle='--', label='Treat None')
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve Analysis - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/10_decision_curve_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def comprehensive_model_evaluation(y_true, y_pred, y_pred_proba, model_name='Model'):
    """
    Perform comprehensive model evaluation
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    """
    print(f"\n🎯 Comprehensive Evaluation: {model_name}")
    print("-" * 80)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['No Disease', 'Disease']))
    
    # Plot all evaluation metrics
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_pred_proba, model_name)
    plot_precision_recall_curve(y_true, y_pred_proba, model_name)
    plot_calibration_curve(y_true, y_pred_proba, model_name)
    plot_decision_curve_analysis(y_true, y_pred_proba, model_name)
    
    print(f"\n  ✓ All evaluation plots for {model_name} saved")


# ================================================================================
# SECTION 9: EXPLAINABLE AI - SHAP ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 9: EXPLAINABLE AI - SHAP ANALYSIS")
print("=" * 80)

def perform_shap_analysis(model, X_train, X_test, feature_names, model_name='Model', 
                          model_type='tree'):
    """
    Perform SHAP analysis for model interpretability
    
    Parameters:
    -----------
    model : object
        Trained model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    model_type : str
        Type of model ('tree' or 'deep')
    """
    print(f"\n🔍 SHAP Analysis: {model_name}")
    print("-" * 80)
    
    # Convert to DataFrame for better visualization
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    
    # Create appropriate explainer
    print("\n  Creating SHAP explainer...")
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
    else:  # deep learning
        # Use a smaller background set for efficiency
        background = shap.sample(X_train_df, 100)
        explainer = shap.DeepExplainer(model, background.values)
        shap_values = explainer.shap_values(X_test_df.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    
    print("  ✓ SHAP values computed")
    
    # 1. Summary Plot (Beeswarm)
    print("\n  Generating SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'outputs/11_shap_summary_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Importance (Bar Plot)
    print("  Generating SHAP feature importance...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/12_shap_importance_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Dependence Plots for top features
    print("  Generating SHAP dependence plots...")
    
    # Get feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-5:]  # Top 5 features
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_features_idx[:6]):
        feature_name = feature_names[idx]
        shap.dependence_plot(idx, shap_values, X_test_df, ax=axes[i], show=False)
        axes[i].set_title(f'Dependence: {feature_name}', fontsize=12, fontweight='bold')
    
    # Hide the last subplot if only 5 features
    if len(top_features_idx) < 6:
        axes[5].axis('off')
    
    plt.suptitle(f'SHAP Dependence Plots - {model_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'outputs/13_shap_dependence_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Waterfall Plot (for first prediction)
    print("  Generating SHAP waterfall plot...")
    plt.figure(figsize=(10, 8))
    
    if model_type == 'tree':
        shap_explainer_v2 = shap.Explainer(model)
        shap_values_v2 = shap_explainer_v2(X_test_df)
        shap.plots.waterfall(shap_values_v2[0], show=False)
    else:
        # For deep learning models, create waterfall manually
        fig, ax = plt.subplots(figsize=(10, 8))
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        
        # Create Explanation object manually
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=X_test_df.iloc[0].values,
                feature_names=feature_names
            ),
            show=False
        )
    
    plt.title(f'SHAP Waterfall Plot (Sample 1) - {model_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'outputs/14_shap_waterfall_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Force Plot (interactive - save as HTML)
    print("  Generating SHAP force plot...")
    
    if model_type == 'tree':
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_test_df.iloc[0],
            matplotlib=True,
            show=False
        )
    else:
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        shap.force_plot(
            expected_value,
            shap_values[0],
            X_test_df.iloc[0],
            matplotlib=True,
            show=False
        )
    
    plt.title(f'SHAP Force Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'outputs/15_shap_force_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n  ✓ SHAP analysis for {model_name} completed")
    
    return shap_values, explainer


def perform_shap_interaction_analysis(model, X_test, feature_names, model_name='Model'):
    """
    Perform SHAP interaction analysis (for tree-based models)
    
    Parameters:
    -----------
    model : object
        Trained tree-based model
    X_test : array-like
        Test features
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    """
    print(f"\n🔄 SHAP Interaction Analysis: {model_name}")
    print("-" * 80)
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print("\n  Computing SHAP interaction values...")
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_test_df)
    
    # Interaction heatmap
    print("  Generating interaction heatmap...")
    interaction_matrix = np.abs(shap_interaction_values).mean(0)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        interaction_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0,
        annot=False,
        fmt='.3f',
        cbar_kws={'label': 'Mean |SHAP Interaction Value|'}
    )
    plt.title(f'SHAP Feature Interaction Heatmap - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/16_shap_interaction_heatmap_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Interaction dependence plots
    print("  Generating interaction dependence plots...")
    
    # Find top interacting pairs
    interaction_strength = np.abs(interaction_matrix)
    np.fill_diagonal(interaction_strength, 0)
    
    # Get top 4 interactions
    top_interactions = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            top_interactions.append((i, j, interaction_strength[i, j]))
    
    top_interactions = sorted(top_interactions, key=lambda x: x[2], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (i, j, strength) in enumerate(top_interactions):
        shap.dependence_plot(
            (i, j),
            shap_interaction_values,
            X_test_df,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(
            f'{feature_names[i]} × {feature_names[j]}\n(Strength: {strength:.4f})',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.suptitle(f'Top SHAP Interaction Dependence Plots - {model_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'outputs/17_shap_interaction_dependence_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  ✓ SHAP interaction analysis completed")
    
    return shap_interaction_values


# ================================================================================
# SECTION 10: EXPLAINABLE AI - LIME ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 10: EXPLAINABLE AI - LIME ANALYSIS")
print("=" * 80)

def perform_lime_analysis(model, X_train, X_test, feature_names, model_type='ml', 
                          num_samples=5):
    """
    Perform LIME analysis for local interpretability
    
    Parameters:
    -----------
    model : object
        Trained model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    feature_names : list
        List of feature names
    model_type : str
        Type of model ('ml' or 'dl')
    num_samples : int
        Number of samples to explain
    """
    print(f"\n🔬 LIME Analysis: Local Interpretability")
    print("-" * 80)
    
    print("\n  Creating LIME explainer...")
    
    # Create prediction function based on model type
    if model_type == 'dl':
        def predict_proba_fn(data):
            preds = model.predict(data, verbose=0)
            return np.hstack([1 - preds, preds])
    else:
        predict_proba_fn = model.predict_proba
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        mode='classification',
        random_state=42
    )
    
    print(f"  Explaining {num_samples} predictions...")
    
    # Explain multiple samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 6*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        exp = explainer.explain_instance(
            X_test[i],
            predict_proba_fn,
            num_features=10
        )
        
        # Create explanation plot
        exp.as_pyplot_figure(label=1)
        plt.title(f'LIME Explanation - Sample {i+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'outputs/18_lime_explanation_sample_{i+1}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ LIME explanations for {num_samples} samples saved")


# ================================================================================
# SECTION 11: PARTIAL DEPENDENCE PLOTS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 11: PARTIAL DEPENDENCE PLOTS")
print("=" * 80)

def create_partial_dependence_plots(model, X_test, feature_names, model_type='dl', 
                                    features_to_plot=None):
    """
    Create partial dependence plots
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : array-like
        Test features
    feature_names : list
        List of feature names
    model_type : str
        Type of model ('ml' or 'dl')
    features_to_plot : list
        List of feature indices/names to plot
    """
    print(f"\n📈 Creating Partial Dependence Plots")
    print("-" * 80)
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Select features to plot
    if features_to_plot is None:
        # Plot top 6 features by importance (if available) or first 6
        features_to_plot = feature_names[:6]
    
    print(f"\n  Generating PDPs for {len(features_to_plot)} features...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        if idx >= 6:
            break
            
        feature_index = feature_names.index(feature) if isinstance(feature, str) else feature
        
        # Create range of values
        values = np.linspace(
            X_test_df.iloc[:, feature_index].min(),
            X_test_df.iloc[:, feature_index].max(),
            50
        )
        
        pdp_values = []
        
        for v in values:
            X_temp = X_test_df.copy()
            X_temp.iloc[:, feature_index] = v
            
            if model_type == 'dl':
                preds = model.predict(X_temp.values, verbose=0).mean()
            else:
                preds = model.predict_proba(X_temp.values)[:, 1].mean()
            
            pdp_values.append(preds)
        
        # Plot
        axes[idx].plot(values, pdp_values, linewidth=2, color='#3498db')
        axes[idx].set_xlabel(feature_names[feature_index], fontsize=11)
        axes[idx].set_ylabel('Predicted Risk', fontsize=11)
        axes[idx].set_title(f'PDP: {feature_names[feature_index]}', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features_to_plot), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Partial Dependence Plots', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('outputs/19_partial_dependence_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Partial dependence plots saved")


# ================================================================================
# SECTION 12: FEATURE CORRELATION ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 12: FEATURE CORRELATION ANALYSIS")
print("=" * 80)

def analyze_feature_correlations(X, feature_names):
    """
    Analyze and visualize feature correlations
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    feature_names : list
        List of feature names
    """
    print(f"\n🔗 Analyzing Feature Correlations")
    print("-" * 80)
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Compute correlation matrix
    corr_matrix = X_df.corr()
    
    # Full correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        annot=False,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/20_correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find highly correlated pairs
    print("\n  Highly correlated feature pairs (|r| > 0.7):")
    high_corr_pairs = []
    
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'Feature 1': feature_names[i],
                    'Feature 2': feature_names[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('Correlation', 
                                                 key=abs, 
                                                 ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("    No highly correlated pairs found")
    
    print("\n  ✓ Correlation analysis completed")


# ================================================================================
# SECTION 13: COMPLETE MODEL COMPARISON
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 13: COMPLETE MODEL COMPARISON")
print("=" * 80)

def create_comprehensive_comparison(ml_results, dl_results, ensemble_results):
    """
    Create comprehensive comparison of all models
    
    Parameters:
    -----------
    ml_results : pandas.DataFrame
        ML model results
    dl_results : pandas.DataFrame
        DL model results
    ensemble_results : pandas.DataFrame
        Ensemble model results
    """
    print(f"\n📊 Creating Comprehensive Model Comparison")
    print("-" * 80)
    
    # Combine all results
    ml_results['Category'] = 'Traditional ML'
    dl_results['Category'] = 'Deep Learning'
    ensemble_results['Category'] = 'Hybrid Ensemble'
    
    # Rename columns for consistency
    if 'Strategy' in ensemble_results.columns:
        ensemble_results = ensemble_results.rename(columns={'Strategy': 'Model'})
    
    all_results = pd.concat([ml_results, dl_results, ensemble_results], 
                            ignore_index=True)
    
    # Sort by accuracy
    all_results = all_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("COMPLETE MODEL COMPARISON - ALL CATEGORIES")
    print("=" * 80)
    print(all_results.to_string(index=False))
    
    # Save to CSV
    all_results.to_csv('outputs/21_complete_model_comparison.csv', 
                       index=False)
    print("\n  ✓ Results saved to CSV")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Top 10 models by accuracy
    top_10 = all_results.head(10)
    colors = ['#e74c3c' if cat == 'Traditional ML' 
              else '#3498db' if cat == 'Deep Learning' 
              else '#2ecc71' for cat in top_10['Category']]
    
    axes[0, 0].barh(range(len(top_10)), top_10['Accuracy'], color=colors)
    axes[0, 0].set_yticks(range(len(top_10)))
    axes[0, 0].set_yticklabels(top_10['Model'])
    axes[0, 0].set_xlabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Top 10 Models by Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Traditional ML'),
        Patch(facecolor='#3498db', label='Deep Learning'),
        Patch(facecolor='#2ecc71', label='Hybrid Ensemble')
    ]
    axes[0, 0].legend(handles=legend_elements, loc='lower right')
    
    # 2. Category-wise performance
    category_avg = all_results.groupby('Category')[['Accuracy', 'Precision', 
                                                     'Recall', 'F1-Score']].mean()
    
    category_avg.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_xlabel('Category', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Average Performance by Category', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. F1-Score vs Accuracy scatter
    for category in all_results['Category'].unique():
        cat_data = all_results[all_results['Category'] == category]
        axes[1, 0].scatter(cat_data['Accuracy'], cat_data['F1-Score'], 
                          label=category, s=100, alpha=0.6)
    
    axes[1, 0].set_xlabel('Accuracy', fontsize=12)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].set_title('Accuracy vs F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 4. Precision vs Recall scatter
    for category in all_results['Category'].unique():
        cat_data = all_results[all_results['Category'] == category]
        axes[1, 1].scatter(cat_data['Recall'], cat_data['Precision'], 
                          label=category, s=100, alpha=0.6)
    
    axes[1, 1].set_xlabel('Recall', fontsize=12)
    axes[1, 1].set_ylabel('Precision', fontsize=12)
    axes[1, 1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/22_comprehensive_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n  ✓ Comprehensive comparison visualizations saved")
    
    return all_results


# ================================================================================
# SECTION 14: MODEL DEPLOYMENT PREPARATION
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 14: MODEL DEPLOYMENT PREPARATION")
print("=" * 80)

def save_best_model(model, scaler, feature_names, model_name='best_model'):
    """
    Save best model and preprocessing objects
    
    Parameters:
    -----------
    model : object
        Best trained model
    scaler : object
        Fitted scaler
    feature_names : list
        List of feature names
    model_name : str
        Name for saved model
    """
    print(f"\n💾 Saving Model for Deployment: {model_name}")
    print("-" * 80)
    
    import joblib
    import json
    
    # Save model
    model_path = f'outputs/{model_name}_model.pkl'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = f'outputs/{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = f'outputs/{model_name}_features.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_names}, f, indent=2)
    print(f"  ✓ Feature names saved: {features_path}")
    
    # Create deployment guide
    guide_path = 'outputs/deployment_guide.txt'
    with open(guide_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HEART DISEASE PREDICTION MODEL - DEPLOYMENT GUIDE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Name: {model_name}\n\n")
        f.write("Files Required for Deployment:\n")
        f.write(f"  1. {model_name}_model.pkl - Trained model\n")
        f.write(f"  2. {model_name}_scaler.pkl - Feature scaler\n")
        f.write(f"  3. {model_name}_features.json - Feature names and order\n\n")
        f.write("Deployment Steps:\n")
        f.write("  1. Load the model: model = joblib.load('{}_model.pkl')\n".format(model_name))
        f.write("  2. Load the scaler: scaler = joblib.load('{}_scaler.pkl')\n".format(model_name))
        f.write("  3. Load feature names: with open('{}_features.json') as f: features = json.load(f)\n".format(model_name))
        f.write("  4. Preprocess input: X_scaled = scaler.transform(X_new)\n")
        f.write("  5. Predict: predictions = model.predict(X_scaled)\n\n")
        f.write("Example Usage:\n")
        f.write("```python\n")
        f.write("import joblib\n")
        f.write("import json\n")
        f.write("import numpy as np\n\n")
        f.write(f"# Load model and scaler\n")
        f.write(f"model = joblib.load('{model_name}_model.pkl')\n")
        f.write(f"scaler = joblib.load('{model_name}_scaler.pkl')\n\n")
        f.write(f"# Prepare new data (ensure features are in correct order)\n")
        f.write(f"new_patient = np.array([[...]])  # Your feature values\n\n")
        f.write(f"# Scale features\n")
        f.write(f"new_patient_scaled = scaler.transform(new_patient)\n\n")
        f.write(f"# Make prediction\n")
        f.write(f"prediction = model.predict(new_patient_scaled)\n")
        f.write(f"probability = model.predict_proba(new_patient_scaled)\n\n")
        f.write(f"print(f'Prediction: {{prediction[0]}}')\n")
        f.write(f"print(f'Disease Probability: {{probability[0][1]:.2%}}')\n")
        f.write("```\n")
    
    print(f"  ✓ Deployment guide saved: {guide_path}")
    print("\n  ✓ All deployment files ready!")


# ================================================================================
# SECTION 15: MAIN EXECUTION PIPELINE
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 15: MAIN EXECUTION PIPELINE")
print("=" * 80)

def main_pipeline(data_path='heart_disease.csv'):
    """
    Execute complete heart disease prediction pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    """
    print("\n" + "=" * 80)
    print("STARTING COMPLETE PIPELINE")
    print("=" * 80)
    
    # NOTE: Uncomment the following sections when you have actual data
    
    
    # 1. Load and explore data
    df = load_and_explore_data(data_path)
    visualize_data_distribution(df, target_column='target_final')
    
    # 2. Preprocess and engineer features
    X, y, feature_names = preprocess_data(df)
    
    # 3. Analyze correlations
    analyze_feature_correlations(X, feature_names)
    
    # 4. Train-test split and scaling
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)
    
    # 5. Train ML models
    ml_models, ml_results = train_ml_models(X_train, X_test, y_train, y_test)
    visualize_ml_results(ml_results)
    
    # 6. Train DL models
    dl_models, dl_results, dl_histories = train_dl_models(X_train, X_test, y_train, y_test)
    visualize_dl_results(dl_results, dl_histories)
    
    # 7. Create hybrid ensemble
    ensemble_results, predictions, meta_model = create_hybrid_ensemble(
        X_train, X_test, y_train, y_test, ml_models, dl_models
    )
    visualize_ensemble_results(ensemble_results)
    
    # 8. Comprehensive model evaluation (best model)
    best_model_name = ensemble_results.iloc[0]['Strategy']
    if 'Stacked' in best_model_name:
        y_pred = predictions['stack_preds']
        y_pred_proba = predictions['stack_probs']
    else:
        y_pred_proba = predictions['xgb_probs']
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    comprehensive_model_evaluation(y_test, y_pred, y_pred_proba, best_model_name)
    
    # 9. SHAP analysis (XGBoost)
    xgb_model = ml_models['XGBoost']
    shap_values, shap_explainer = perform_shap_analysis(
        xgb_model, X_train, X_test, feature_names, 
        'XGBoost', model_type='tree'
    )
    
    # 10. SHAP interaction analysis
    shap_interaction_values = perform_shap_interaction_analysis(
        xgb_model, X_test, feature_names, 'XGBoost'
    )
    
    # 11. LIME analysis
    perform_lime_analysis(
        xgb_model, X_train, X_test, feature_names, 
        model_type='ml', num_samples=3
    )
    
    # 12. Partial dependence plots
    create_partial_dependence_plots(
        xgb_model, X_test, feature_names, 
        model_type='ml', features_to_plot=feature_names[:6]
    )
    
    # 13. Comprehensive comparison
    all_results = create_comprehensive_comparison(ml_results, dl_results, ensemble_results)
    
    # 14. Save best model
    if 'Stacked' in ensemble_results.iloc[0]['Strategy']:
        save_best_model(meta_model, scaler, ['XGB_prob', 'DL_prob'], 'stacked_hybrid')
    else:
        save_best_model(xgb_model, scaler, feature_names, 'xgboost')
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 All outputs saved to: /mnt/user-data/outputs/")
    print("✓ Total visualizations: 22+")
    print("✓ Model comparison CSV: complete_model_comparison.csv")
    print("✓ Deployment files: model.pkl, scaler.pkl, features.json")
    print("\n🎉 Project Complete!")
    
    
    print("\n" + "=" * 80)
    print("PIPELINE TEMPLATE READY")
    print("=" * 80)
    print("\nTo run the pipeline:")
    print("1. Replace 'heart_disease.csv' with your actual data file path")
    print("2. Uncomment the code in the main_pipeline() function")
    print("3. Adjust feature engineering based on your column names")
    print("4. Run: main_pipeline('your_data.csv')")
    print("=" * 80)
  

    print("\n💾 Saving base models...")
    # Save XGBoost model
    xgb_model = ml_models['XGBoost']
    joblib.dump(xgb_model, 'outputs/xgboost_model.pkl')
    print("  ✓ XGBoost saved")

# Save best DL model (choose one)
    best_dl_model = dl_models[list(dl_models.keys())[0]]  # or pick best manually
    best_dl_model.save('outputs/dl_model.h5')
    print("  ✓ Deep Learning model saved")



# ==========================================
# 🔥 SAVE BASE MODELS (IMPORTANT)
# ==========================================


import json

# Save stacking feature names
with open('outputs/stack_features.json', 'w') as f:
    json.dump(['XGB_prob', 'DL_prob'], f)

print("  ✓ Stack features saved")
# ================================================================================
# RUN THE PIPELINE
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HEART DISEASE PREDICTION - EXPLAINABLE AI")
    print("Complete Implementation Ready")
    print("=" * 80)
    
    # Uncomment to run with your data:
    main_pipeline('heart_disease.csv')
    
    print("\n✓ Script loaded successfully!")
    print("✓ All functions defined and ready to use")
    print("\nNext steps:")
    print("  1. Load your dataset")
    print("  2. Uncomment the main_pipeline() call")
    print("  3. Execute the script")
    print("\n" + "=" * 80)
