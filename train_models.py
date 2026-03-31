import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.calibration import calibration_curve
import json
import pickle
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = "breast_cancer.csv"
TARGET = "diagnosis"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODEL_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set visual style
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

# ================= LOAD AND PREPARE DATA =================
def load_data():
    """Load and prepare the dataset"""
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        
        # Check if target column exists
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in dataset")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df[TARGET].value_counts()}")
        
        # Encode target: M=1 (malignant), B=0 (benign)
        y = df[TARGET].map({"B": 0, "M": 1})
        X = df.drop(columns=[TARGET])
        feature_names = X.columns.tolist()
        
        logger.info(f"Number of features: {len(feature_names)}")
        
        return X, y, feature_names
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# ================= TRAIN MODELS =================
def train_models(X_train, y_train, X_test, y_test, X_scaled, y):
    """Train and evaluate multiple models"""
    models = {
        "svm": SVC(kernel="linear", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    metrics = {}
    trained_models = {}
    
    # Use stratified k-fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1")
            
            metrics[name] = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1_score": f1_score(y_test, y_pred),
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "cv_f1_scores": cv_scores.tolist(),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Add ROC AUC if model supports probability predictions
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                metrics[name]["roc_auc"] = roc_auc
                metrics[name]["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            
            logger.info(f"{name} - Accuracy: {metrics[name]['test_accuracy']:.4f}, F1: {metrics[name]['test_f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    return trained_models, metrics

# ================= VISUALIZATIONS =================
def create_visualizations(models, metrics, X_test, y_test, feature_names):
    """Create comprehensive visualizations"""
    
    # 1. Model Comparison Plot
    plt.figure(figsize=(12, 8))
    model_names = list(metrics.keys())
    accuracy_scores = [metrics[name]['test_accuracy'] for name in model_names]
    f1_scores = [metrics[name]['test_f1_score'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [name.replace('_', ' ').title() for name in model_names])
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(accuracy_scores):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(f1_scores):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Random Forest Feature Importance
    if 'random_forest' in models:
        rf_model = models['random_forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], ax=axes[0])
        axes[0].set_title('Feature Importance (Random Forest)')
        axes[0].set_xlabel('Importance')
    
    # Logistic Regression Coefficients
    if 'logistic_regression' in models:
        log_reg = models['logistic_regression']
        coefficients = log_reg.coef_[0]
        indices = np.argsort(np.abs(coefficients))[-15:]  # Top 15 features by absolute value
        
        sns.barplot(x=coefficients[indices], y=np.array(feature_names)[indices], ax=axes[1])
        axes[1].set_title('Feature Coefficients (Logistic Regression)')
        axes[1].set_xlabel('Coefficient Value')
        axes[1].axvline(0, color='black', linewidth=1)
    
    # SVM Coefficients (Linear Kernel)
    if 'svm' in models:
        svm_model = models['svm']
        svm_coeffs = svm_model.coef_[0]
        indices = np.argsort(np.abs(svm_coeffs))[-15:]  # Top 15 features by absolute value
        
        sns.barplot(x=svm_coeffs[indices], y=np.array(feature_names)[indices], ax=axes[2])
        axes[2].set_title('Feature Coefficients (SVM - Linear Kernel)')
        axes[2].set_xlabel('Coefficient Value')
        axes[2].axvline(0, color='black', linewidth=1)
    
    # 3. ROC Curve
    axes[3].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    for name in models:
        if 'roc_curve' in metrics.get(name, {}):
            fpr = metrics[name]['roc_curve']['fpr']
            tpr = metrics[name]['roc_curve']['tpr']
            roc_auc = metrics[name]['roc_auc']
            axes[3].plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
    
    axes[3].set_xlabel('False Positive Rate')
    axes[3].set_ylabel('True Positive Rate')
    axes[3].set_title('ROC Curves')
    axes[3].legend(loc='lower right')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, ax) in enumerate(zip(models.keys(), axes)):
        if name in metrics:
            cm = np.array(metrics[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Benign', 'Malignant'],
                       yticklabels=['Benign', 'Malignant'])
            ax.set_title(f'{name.replace("_", " ").title()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ================= SAVE RESULTS =================
def save_results(models, metrics, scaler, feature_names):
    """Save models, metrics, and other artifacts"""
    
    # Save models
    for name, model in models.items():
        with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), "wb") as f:
            pickle.dump(model, f)
    
    # Save scaler
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    
    # Save metrics with additional metadata
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "path": DATA_PATH,
            "target": TARGET,
            "feature_count": len(feature_names)
        },
        "metrics": metrics,
        "training_parameters": {
            "test_size": 0.3,
            "random_state": 42,
            "cv_folds": 5
        }
    }
    
    with open(os.path.join(MODEL_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Save a simplified version for the web app
    simplified_metrics = {}
    for name, metric in metrics.items():
        simplified_metrics[name] = {
            "accuracy": round(metric["test_accuracy"], 4),
            "precision": round(metric["test_precision"], 4),
            "recall": round(metric["test_recall"], 4),
            "f1_score": round(metric["test_f1_score"], 4),
            "roc_auc": round(metric.get("roc_auc", 0), 4)
        }
    
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(simplified_metrics, f, indent=4)

# ================= COPY TO STATIC FOLDER =================
def copy_to_static():
    """Copy visualization files to static folder"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    plots = ["model_comparison.png", "feature_analysis.png", "confusion_matrices.png"]
    
    for plot_name in plots:
        src = os.path.join(MODEL_DIR, plot_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(static_dir, plot_name))
            logger.info(f"Copied {plot_name} to static folder")

# ================= MAIN EXECUTION =================
def main():
    """Main training function"""
    try:
        logger.info("=" * 50)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 50)
        
        # Load data
        X, y, feature_names = load_data()
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train models
        models, metrics = train_models(X_train, y_train, X_test, y_test, X_scaled, y)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        create_visualizations(models, metrics, X_test, y_test, feature_names)
        
        # Save results
        logger.info("Saving results...")
        save_results(models, metrics, scaler, feature_names)
        
        # Copy to static folder
        copy_to_static()
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        for name, metric in metrics.items():
            logger.info(f"{name.replace('_', ' ').title():<20} | Accuracy: {metric['test_accuracy']:.4f} | F1: {metric['test_f1_score']:.4f} | AUC: {metric.get('roc_auc', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)