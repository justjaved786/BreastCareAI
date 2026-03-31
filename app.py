from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import numpy as np
import pickle
import io
import os
import re
import json
import secrets
import pandas as pd
import sqlite3
from fpdf import FPDF
import logging
from functools import wraps
from datetime import datetime, timedelta
from jinja2 import TemplateNotFound
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
from datetime import datetime
from image_model.image_utils import preprocess_image, predict_image

# Initialize Flask application
app = Flask(__name__)

# 🔐 FIXED SECRET KEY (stable across restarts)
app.secret_key = "breastcareai_super_secret_key_2025"

# ================= SESSION CONFIGURATION =================
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
    'FLASK_ENV', 'development'
) == 'production'
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_NAME"] = "breastcareai_session"

# ❗ IMPORTANT: prevent session rewrite on every request
app.config["SESSION_REFRESH_EACH_REQUEST"] = False

# ================= LOGGING =================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========= Database Configuration =========
DATABASE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with users table and user history table"""
    conn = get_db_connection()
    
    # Users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date_of_birth DATE,
            gender TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    
    # User history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,
            model_used TEXT,
            result TEXT,
            confidence REAL,
            input_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create indexes for better performance
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_history_user_id 
        ON user_history (user_id)
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_history_created_at 
        ON user_history (created_at)
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ========= User Management Functions =========
def register_user(name, email, password, date_of_birth=None, gender=None):
    """Register a new user - role removed from parameters"""
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (name, email, password_hash, date_of_birth, gender, role) VALUES (?, ?, ?, ?, ?, ?)',
            (name, email, password_hash, date_of_birth, gender, 'user')  # Role hardcoded to 'user'
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        raise Exception("Email already exists")
    except Exception as e:
        raise Exception(f"Registration failed: {str(e)}")
    finally:
        conn.close()

def validate_login(email, password):
    """Validate user login credentials"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            return dict(user)
        return None
    except Exception as e:
        logger.error(f"Login validation error: {str(e)}")
        return None
    finally:
        conn.close()

def check_email_exists(email):
    """Check if email exists in database"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()

def reset_user_password(email, new_password):
    """Reset user password"""
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(new_password)
        conn.execute(
            'UPDATE users SET password_hash = ? WHERE email = ?',
            (password_hash, email)
        )
        conn.commit()
        return conn.total_changes > 0
    except Exception as e:
        raise Exception(f"Password reset failed: {str(e)}")
    finally:
        conn.close()

def get_user_by_email(email):
    """Get user by email"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()

def update_user_profile(user_id, name, date_of_birth, gender, phone, address):
    """Update user profile information"""
    conn = get_db_connection()
    try:
        conn.execute(
            '''UPDATE users 
               SET name = ?, date_of_birth = ?, gender = ?, phone = ?, address = ?
               WHERE id = ?''',
            (name, date_of_birth, gender, phone, address, user_id)
        )
        conn.commit()
        return conn.total_changes > 0
    except Exception as e:
        raise Exception(f"Profile update failed: {str(e)}")
    finally:
        conn.close()

def delete_user_account(user_id):
    """Delete user account"""
    conn = get_db_connection()
    try:
        # Delete user history first
        conn.execute('DELETE FROM user_history WHERE user_id = ?', (user_id,))
        # Then delete user
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        return conn.total_changes > 0
    except Exception as e:
        raise Exception(f"Account deletion failed: {str(e)}")
    finally:
        conn.close()

def add_user_history(user_id, prediction_type, model_used=None, result=None, confidence=None, input_data=None):
    """Add a record to user history"""
    conn = get_db_connection()
    try:
        # Convert input_data to JSON string if it's a dictionary
        if isinstance(input_data, dict):
            input_data = json.dumps(input_data)
        
        conn.execute(
            '''INSERT INTO user_history 
               (user_id, prediction_type, model_used, result, confidence, input_data) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (user_id, prediction_type, model_used, result, confidence, input_data)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error adding user history: {str(e)}")
        return False
    finally:
        conn.close()

def get_user_history(user_id, limit=50):
    """Get user prediction history"""
    conn = get_db_connection()
    try:
        history = conn.execute(
            '''SELECT * FROM user_history 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?''',
            (user_id, limit)
        ).fetchall()
        return [dict(item) for item in history]
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        return []
    finally:
        conn.close()

def get_user_stats(user_id):
    """Get user statistics for profile page"""
    conn = get_db_connection()
    try:
        # Total predictions
        total_preds = conn.execute(
            'SELECT COUNT(*) as count FROM user_history WHERE user_id = ?',
            (user_id,)
        ).fetchone()['count']
        
        # Predictions by type
        type_stats = conn.execute(
            '''SELECT prediction_type, COUNT(*) as count 
               FROM user_history 
               WHERE user_id = ? 
               GROUP BY prediction_type''',
            (user_id,)
        ).fetchall()
        
        # Most used model
        model_stats = conn.execute(
            '''SELECT model_used, COUNT(*) as count 
               FROM user_history 
               WHERE user_id = ? AND model_used IS NOT NULL
               GROUP BY model_used 
               ORDER BY count DESC 
               LIMIT 1''',
            (user_id,)
        ).fetchone()
        
        # Recent activity
        recent_activity = conn.execute(
            '''SELECT prediction_type, created_at 
               FROM user_history 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT 5''',
            (user_id,)
        ).fetchall()
        
        return {
            'total_predictions': total_preds,
            'predictions_by_type': {row['prediction_type']: row['count'] for row in type_stats},
            'most_used_model': model_stats['model_used'] if model_stats else 'None',
            'recent_activity': [dict(row) for row in recent_activity]
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return {}
    finally:
        conn.close()

# ========= Model Configuration =========
MODEL_DIR = "model"
IMAGE_MODEL_DIR = "image_models"

# Registry of models and their display names
MODEL_REGISTRY = {
    "svm": {
        "path": os.path.join(MODEL_DIR, "svm_model.pkl"),
        "display": "Support Vector Machine"
    },
    "random_forest": {
        "path": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
        "display": "Random Forest"
    },
    "logistic_regression": {
        "path": os.path.join(MODEL_DIR, "logistic_regression_model.pkl"),
        "display": "Logistic Regression"
    },
}



# 8 features to match index.html
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
]

# Feature importance data
FEATURE_IMPORTANCE = {
    'radius_mean': 0.28,
    'concave_points_mean': 0.22,
    'perimeter_mean': 0.18,
    'concavity_mean': 0.12,
    'area_mean': 0.08,
    'compactness_mean': 0.06,
    'texture_mean': 0.04,
    'smoothness_mean': 0.02
}

# Typical value ranges for benign and malignant cases
BENIGN_AVERAGES = {
    'radius_mean': 12.15,
    'texture_mean': 17.92,
    'perimeter_mean': 78.82,
    'area_mean': 462.8,
    'smoothness_mean': 0.0925,
    'compactness_mean': 0.0801,
    'concavity_mean': 0.0461,
    'concave_points_mean': 0.0257
}

MALIGNANT_AVERAGES = {
    'radius_mean': 17.46,
    'texture_mean': 21.60,
    'perimeter_mean': 115.37,
    'area_mean': 978.38,
    'smoothness_mean': 0.1029,
    'compactness_mean': 0.1452,
    'concavity_mean': 0.1608,
    'concave_points_mean': 0.0874
}

# Symptoms assessment questions
SYMPTOMS_QUESTIONS = [
    {
        'id': 'lump',
        'question': 'Have you noticed any lumps or thickening in your breast or underarm area?',
        'type': 'yes_no'
    },
    {
        'id': 'pain',
        'question': 'Do you experience persistent pain in your breast or armpit?',
        'type': 'yes_no'
    },
    {
        'id': 'nipple_changes',
        'question': 'Have you noticed any changes in your nipples (inversion, discharge, rash)?',
        'type': 'yes_no'
    },
    {
        'id': 'skin_changes',
        'question': 'Have you observed any changes in your breast skin (redness, dimpling, puckering)?',
        'type': 'yes_no'
    },
    {
        'id': 'swelling',
        'question': 'Do you have any swelling in your breast or armpit?',
        'type': 'yes_no'
    },
    {
        'id': 'family_history',
        'question': 'Do you have a family history of breast cancer (mother, sister, daughter)?',
        'type': 'yes_no'
    },
    {
        'id': 'age',
        'question': 'What is your age group?',
        'type': 'multiple_choice',
        'options': ['Under 30', '30-39', '40-49', '50-59', '60 and above']
    },
    {
        'id': 'menstrual_history',
        'question': 'When did you have your first menstrual period?',
        'type': 'multiple_choice',
        'options': ['Under 12 years', '12-13 years', '14 years and above']
    },
    {
        'id': 'menopause',
        'question': 'Have you reached menopause?',
        'type': 'yes_no'
    },
    {
        'id': 'hormone_therapy',
        'question': 'Have you ever used hormone replacement therapy?',
        'type': 'yes_no'
    }
]

# ========= Load artifacts =========
try:
    if not os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
        raise FileNotFoundError("Scaler file not found")
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    MODELS = {}
    for k, v in MODEL_REGISTRY.items():
        if not os.path.exists(v["path"]):
            logger.warning(f"Model file not found: {v['path']}")
            continue
        try:
            MODELS[k] = pickle.load(open(v["path"], "rb"))
            logger.info(f"Successfully loaded model: {k}")
        except Exception as model_error:
            logger.error(f"Error loading model {k}: {str(model_error)}")
            continue
    
    if not MODELS:
        logger.error("No models were successfully loaded")
        raise Exception("No models available for predictions")
        
    logger.info(f"Successfully loaded {len(MODELS)} models and scaler")
    
except Exception as e:
    logger.error(f"Error loading models or scaler: {str(e)}")
    MODELS = {}
    scaler = None



# Optional metrics file
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
METRICS = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            METRICS = json.load(f)
        logger.info("Successfully loaded model metrics")
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")

# ========= Decorators =========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            logger.warning("User not logged in, redirecting to login")
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def models_loaded_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not MODELS or scaler is None:
            logger.error("Models or scaler not loaded")
            flash(
                "Prediction models are not available. Please check model files and contact administrator.",
                "danger"
            )
            return redirect(url_for('common_page'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in as admin.", "warning")
            return redirect(url_for('login'))

        user_data = get_user_by_id(session["user_id"])
        if not user_data or user_data.get("role") != "admin":
            flash("Access denied. Admin privileges required.", "danger")
            return redirect(url_for('common_page'))

        return f(*args, **kwargs)
    return decorated_function

# ========= Helper functions =========
def get_selected_models_from_form(form) -> list:
    """Read <select multiple name='model_name'> and normalize to our model keys."""
    selected = form.getlist("model_name")
    logger.debug(f"Form models selected: {selected}")
    if not selected or "all" in selected:
        logger.info("No specific models selected or 'all' chosen, using all models")
        return list(MODELS.keys())
    valid_models = [m for m in selected if m in MODELS]
    if not valid_models:
        raise ValueError("Please select at least one model")
    return valid_models

def build_feature_row_from_form(form) -> np.ndarray:
    values, missing, invalid = [], [], []
    logger.debug(f"Form inputs: {form.to_dict()}")
    for name in FEATURE_NAMES:
        raw = form.get(name, "").strip()
        logger.debug(f"Processing {name}: {raw}")
        if raw == "":
            missing.append(name)
            continue
        try:
            val = float(raw)
            values.append(val)
        except ValueError:
            invalid.append(name)
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    if invalid:
        raise ValueError(f"Invalid numeric values for: {', '.join(invalid)}")
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("Incomplete feature set provided")
    logger.debug(f"Feature row: {values}")
    return np.array(values, dtype=float).reshape(1, -1)

def map_proba_to_benign_malignant(model, proba_row: np.ndarray) -> tuple:
    """Return (p_benign, p_malignant) consistently, respecting model.classes_ order."""
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if len(classes) == 2:
            try:
                idx_benign = classes.index(0)
                idx_malignant = classes.index(1)
            except ValueError:
                idx_benign, idx_malignant = 0, 1
        else:
            idx_benign, idx_malignant = 0, 1
    else:
        idx_benign, idx_malignant = 0, 1
    p_benign = float(proba_row[idx_benign])
    p_malignant = float(proba_row[idx_malignant])
    return p_benign, p_malignant

def ensure_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probabilities shape (n,2). Fallback to decision_function if needed."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        elif proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X).astype(float)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
    preds = model.predict(X)
    out = np.zeros((len(preds), 2), dtype=float)
    for i, y in enumerate(preds):
        if int(y) == 1:
            out[i, 0] = 0.0; out[i, 1] = 1.0
        else:
            out[i, 0] = 1.0; out[i, 1] = 0.0
    return out

def make_results_for_single_sample(X_scaled: np.ndarray, selected_keys: list):
    """Returns dict keyed by model key with model count and prediction details."""
    results = {}
    model_count = len(selected_keys)
    model_selection_text = f"{model_count} Model{'s' if model_count != 1 else ''} Selected" if model_count < len(MODELS) else "All Models Selected"
    for key in selected_keys:
        model = MODELS[key]
        display = MODEL_REGISTRY[key]["display"]
        proba = ensure_proba(model, X_scaled)
        p_benign, p_malignant = map_proba_to_benign_malignant(model, proba[0])
        predicted_label = "Malignant" if p_malignant >= p_benign else "Benign"
        confidence = float(max(p_benign, p_malignant))
        
        # Handle accuracy more robustly
        accuracy = None
        if key in METRICS and isinstance(METRICS[key], dict):
            for acc_key in ["test_accuracy", "accuracy", "train_accuracy", "val_accuracy"]:
                if acc_key in METRICS[key]:
                    accuracy = float(METRICS[key][acc_key])
                    break
        
        results[key] = {
            "display": display,
            "predicted_label": predicted_label,
            "proba": {"benign": p_benign, "malignant": p_malignant},
            "confidence": confidence,
            "accuracy": accuracy
        }
    return results, model_selection_text

def pick_best_for_pdf(results: dict):
    """Pick the model with highest confidence for PDF summary."""
    best_key = max(results.keys(), key=lambda k: results[k]["confidence"])
    item = results[best_key]
    return {
        "model": item["display"],
        "prediction": "Breast Cancer Detected" if item["predicted_label"] == "Malignant" else "No Cancer Detected",
        "malignant": item["proba"]["malignant"],
        "benign": item["proba"]["benign"],
    }

def get_precautions(malignant_probability):
    """Return appropriate precautions based on malignancy probability."""
    if malignant_probability > 0.7:
        return [
            "Consult an oncologist or breast specialist immediately.",
            "Schedule a diagnostic mammogram and ultrasound.",
            "Consider a biopsy for definitive diagnosis.",
            "Discuss treatment options with healthcare providers.",
            "Seek support from cancer care organizations."
        ]
    elif malignant_probability > 0.4:
        return [
            "Consult with a healthcare professional for further evaluation.",
            "Schedule follow-up imaging in 3-6 months.",
            "Consider additional diagnostic tests if recommended.",
            "Maintain regular breast self-exams.",
            "Follow up with your primary care physician."
        ]
    else:
        return None

def get_recommendations(risk_level, is_malignant):
    """Return recommendations based on risk level and prediction."""
    if is_malignant:
        return [
            "Consult with an oncologist or breast specialist as soon as possible.",
            "Schedule a follow-up diagnostic mammogram or ultrasound.",
            "Consider a biopsy for definitive diagnosis if recommended by your doctor.",
            "Discuss treatment options with your healthcare provider.",
            "Seek support from cancer support groups or counselors."
        ]
    elif risk_level > 0.3:
        return [
            "Schedule a follow-up appointment with your doctor to discuss these results.",
            "Consider additional screening such as a diagnostic mammogram or ultrasound.",
            "Monitor for any changes in your breast health and report them to your doctor.",
            "Maintain a healthy lifestyle with regular exercise and a balanced diet.",
            "Continue with regular breast cancer screenings as recommended for your age group."
        ]
    else:
        return [
            "Continue with regular breast self-exams monthly.",
            "Schedule routine mammograms as recommended for your age group.",
            "Maintain a healthy lifestyle with regular exercise and a balanced diet.",
            "Be aware of any changes in your breast health and report them to your doctor.",
            "Discuss your family history with your doctor to assess any genetic risks."
        ]

def get_detailed_recommendations(prediction_result, malignant_probability, user_age=None, user_gender=None):
    """Get detailed recommendations for PDF report based on prediction result and user profile."""
    recommendations = []
    
    if prediction_result == "Malignant" or malignant_probability > 0.7:
        recommendations.extend([
            "IMMEDIATE MEDICAL ATTENTION REQUIRED:",
            "- Schedule an appointment with an oncologist or breast specialist within 1-2 weeks",
            "- Request diagnostic mammography and breast ultrasound",
            "- Discuss biopsy options with your healthcare provider",
            "- Consider genetic counseling if you have family history",
            "- Explore treatment options: surgery, chemotherapy, radiation, or targeted therapy"
        ])
    elif malignant_probability > 0.4:
        recommendations.extend([
            "MODERATE RISK - FURTHER EVALUATION RECOMMENDED:",
            "- Schedule follow-up with primary care physician within 1 month",
            "- Consider diagnostic imaging within 3-6 months",
            "- Monitor for any changes in breast tissue",
            "- Maintain regular breast self-examinations",
            "- Discuss risk factors with healthcare provider"
        ])
    else:
        recommendations.extend([
            "LOW RISK - PREVENTIVE CARE RECOMMENDATIONS:",
            "- Continue monthly breast self-examinations",
            "- Schedule routine screening mammograms as per age guidelines",
            "- Maintain healthy lifestyle with balanced diet and regular exercise",
            "- Limit alcohol consumption and avoid smoking",
            "- Maintain healthy body weight"
        ])
    
    # Age-specific recommendations
    if user_age:
        try:
            age = int(user_age)
            if age < 40:
                recommendations.extend([
                    "",
                    "AGE-SPECIFIC RECOMMENDATIONS (Under 40):",
                    "- Clinical breast exam every 3 years",
                    "- Discuss breast awareness and self-examination techniques",
                    "- Consider baseline mammogram if high risk factors present"
                ])
            elif 40 <= age <= 49:
                recommendations.extend([
                    "",
                    "AGE-SPECIFIC RECOMMENDATIONS (40-49):",
                    "- Annual screening mammograms",
                    "- Discuss personalized screening schedule with doctor",
                    "- Consider additional screening if dense breast tissue"
                ])
            else:
                recommendations.extend([
                    "",
                    "AGE-SPECIFIC RECOMMENDATIONS (50+):",
                    "- Continue annual or biennial mammograms",
                    "- Discuss benefits and risks of continued screening",
                    "- Consider overall health status in screening decisions"
                ])
        except (ValueError, TypeError):
            pass
    
    # Gender-specific recommendations
    if user_gender and user_gender.lower() == 'male':
        recommendations.extend([
            "",
            "MALE-SPECIFIC RECOMMENDATIONS:",
            "- Be aware that breast cancer can occur in men",
            "- Report any breast changes, lumps, or pain to your doctor",
            "- Discuss family history of breast cancer with your physician"
        ])
    
    return recommendations

def calculate_overall_confidence(results):
    """Calculate overall confidence from multiple models."""
    malignant_results = [r for r in results.values() if r["predicted_label"] == "Malignant"]
    benign_results = [r for r in results.values() if r["predicted_label"] == "Benign"]
    
    malignant_confidence = 0
    benign_confidence = 0
    
    if malignant_results:
        malignant_confidence = sum(r["confidence"] for r in malignant_results) / len(malignant_results)
    
    if benign_results:
        benign_confidence = sum(r["confidence"] for r in benign_results) / len(benign_results)
    
    if malignant_results and benign_results:
        total_accuracy = sum(r.get("accuracy", 0.7) for r in results.values())
        
        malignant_confidence = sum(
            r["confidence"] * (r.get("accuracy", 0.7) / total_accuracy) 
            for r in malignant_results
        )
        
        benign_confidence = sum(
            r["confidence"] * (r.get("accuracy", 0.7) / total_accuracy) 
            for r in benign_results
        )
    
    total = malignant_confidence + benign_confidence
    if total > 0:
        malignant_confidence /= total
        benign_confidence /= total
    
    return benign_confidence, malignant_confidence



def calculate_symptoms_risk(answers):
    """Calculate risk based on symptoms assessment"""
    risk_score = 0
    risk_factors = []
    
    # Yes/No questions scoring
    yes_no_questions = ['lump', 'pain', 'nipple_changes', 'skin_changes', 'swelling', 'family_history', 'menopause', 'hormone_therapy']
    for question in yes_no_questions:
        if answers.get(question) == 'yes':
            risk_score += 2
            risk_factors.append(question)
    
    # Age group scoring
    age_group = answers.get('age', '')
    if age_group == '40-49':
        risk_score += 1
    elif age_group == '50-59':
        risk_score += 2
    elif age_group == '60 and above':
        risk_score += 3
    
    # Menstrual history scoring
    menstrual_history = answers.get('menstrual_history', '')
    if menstrual_history == 'Under 12 years':
        risk_score += 1
    
    # Calculate risk level
    max_score = 19  # Maximum possible score
    risk_percentage = (risk_score / max_score) * 100
    
    # Determine risk category
    if risk_percentage < 20:
        risk_category = "Low Risk"
        color = "green"
    elif risk_percentage < 50:
        risk_category = "Moderate Risk"
        color = "orange"
    else:
        risk_category = "High Risk"
        color = "red"
    
    return {
        'risk_score': risk_score,
        'risk_percentage': risk_percentage,
        'risk_category': risk_category,
        'color': color,
        'risk_factors': risk_factors
    }

def sanitize_text_for_pdf(text):
    """Enhanced sanitize text for PDF generation to handle encoding issues"""
    if not text:
        return ""
    
    # Replace common problematic Unicode characters with ASCII equivalents
    replacements = {
        '•': '-',    # Bullet to hyphen
        '\u2022': '-', # Unicode bullet to hyphen
        '–': '-',    # En dash to hyphen
        '—': '-',    # Em dash to hyphen
        '‘': "'",    # Left single quote to apostrophe
        '’': "'",    # Right single quote to apostrophe
        '“': '"',    # Left double quote to quote
        '”': '"',    # Right double quote to quote
        '…': '...',  # Ellipsis
        '●': '-',    # Black circle bullet to hyphen
        '▪': '-',    # Black small square to hyphen
        '■': '-',    # Black square to hyphen
        '→': '->',   # Right arrow
        '←': '<-',   # Left arrow
        '±': '+/-',  # Plus-minus
        '×': 'x',    # Multiplication sign
        '÷': '/',    # Division sign
        '°': ' deg', # Degree symbol
        'µ': 'u',    # Micro symbol
        'α': 'alpha',# Greek alpha
        'β': 'beta', # Greek beta
        'γ': 'gamma',# Greek gamma
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any other non-ASCII characters and replace with closest ASCII or remove
    text = text.encode('ascii', 'replace').decode('ascii')
    
    # Replace the replacement character (?) with empty string if it appears
    text = text.replace('?', '')
    
    return text

# ========= Enhanced PDF Generation Class with Breast Cancer Awareness Theme =========
class BreastCareAIPDFReport(FPDF):
    """Enhanced PDF report with BreastCareAI branding and breast cancer awareness theme"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        self.left_margin = 15
        self.right_margin = 15
        self.top_margin = 35  # Increased for larger header
        self.bottom_margin = 25
        self.set_margins(self.left_margin, self.top_margin, self.right_margin)
        
        # Breast Cancer Awareness Color Scheme
        self.colors = {
            # Primary brand colors
            'pink': (236, 72, 153),        # #EC4899 - Primary pink
            'navy': (10, 31, 68),          # #0A1F44 - Primary navy
            'soft_blue': (37, 99, 235),    # #2563EB - Tech accent blue
            
            # Background and supporting colors
            'light_pink': (253, 242, 248), # Very light pink for backgrounds
            'light_blue': (239, 246, 255), # Light blue for accents
            'medium_gray': (107, 114, 128),# Medium gray for secondary text
            'light_gray': (243, 244, 246), # Light gray for backgrounds
            
            # Status colors
            'success': (16, 185, 129),     # Green for benign/low risk
            'warning': (245, 158, 11),     # Orange for moderate risk
            'danger': (239, 68, 68),       # Red for malignant/high risk
            
            # Text colors
            'dark_text': (17, 24, 39),     # Dark text
            'medium_text': (75, 85, 99),   # Medium text
            'light_text': (255, 255, 255), # Light text for dark backgrounds
        }
        
        # Report metadata
        self.report_id = secrets.token_hex(8).upper()
        self.creation_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
        self.page_count = 0
        
    def header(self):
        """Fixed header with BreastCareAI branding"""
        # Header background with brand colors
        self.set_fill_color(*self.colors['navy'])
        self.rect(0, 0, 210, 32, 'F')
        
        # Pink accent strip
        self.set_fill_color(*self.colors['pink'])
        self.rect(0, 32, 210, 3, 'F')
        
        # Main title - Centered and prominent
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*self.colors['light_text'])
        self.set_xy(0, 12)
        self.cell(0, 8, "BreastCareAI | AI-BASED ANALYSIS REPORT", 0, 0, 'C')
        
        # Page number with subtle styling
        self.set_font('Arial', 'I', 9)
        self.set_text_color(*self.colors['light_text'])
        self.set_xy(-25, 15)
        self.cell(0, 6, f"Page {self.page_no()}", 0, 0, 'R')
        
        self.set_y(40)  # Reset Y position below header
    
    def footer(self):
        """Fixed footer with BreastCareAI branding"""
        # Position at 1.5 cm from bottom
        self.set_y(-20)
        
        # Footer background
        self.set_fill_color(*self.colors['navy'])
        self.rect(0, 277, 210, 20, 'F')
        
        # Pink accent strip at top of footer
        self.set_fill_color(*self.colors['pink'])
        self.rect(0, 277, 210, 2, 'F')
        
        # Footer content - Centered "Generated by BreastCareAI"
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*self.colors['light_text'])
        
        # Center - Generated by BreastCareAI
        self.set_xy(0, 282)
        self.cell(0, 6, "Generated by BreastCareAI", 0, 0, 'C')
        
        # Bottom line with report info
        self.set_font('Arial', '', 7)
        self.set_xy(0, 288)
        self.cell(0, 4, f"Report ID: {self.report_id} | Generated: {self.creation_time}", 0, 0, 'C')
    
    def create_cover_page(self, user_data, patient_info):
        """Create professional cover page with breast cancer awareness theme"""
        self.add_page()
        
        # Background with light pink - breast cancer awareness theme
        self.set_fill_color(*self.colors['light_pink'])
        self.rect(0, 0, 210, 297, 'F')
        
        # Add a decorative header bar with brand colors
        self.set_fill_color(*self.colors['navy'])
        self.rect(0, 0, 210, 60, 'F')
        
        # Pink accent strip
        self.set_fill_color(*self.colors['pink'])
        self.rect(0, 60, 210, 4, 'F')
        
        # Main title - Centered and prominent
        self.set_font('Arial', 'B', 24)
        self.set_text_color(*self.colors['light_text'])
        self.set_xy(0, 25)
        self.cell(0, 12, "BreastCareAI", 0, 1, 'C')
        
        self.set_font('Arial', 'B', 20)
        self.cell(0, 12, "AI-BASED ANALYSIS REPORT", 0, 1, 'C')
        
        # Motivational Subtitle - Enhanced Look
        self.set_font('Arial', 'BI', 15)  # Bold Italic = strong yet elegant
        self.set_text_color(233, 30, 99)  # Brighter, confident pink
        self.set_xy(0, 95)  # Slightly lower for breathing space
        self.cell(0, 12, "Exploring Insights Through AI", 0, 1, 'C')

        
        # Add a decorative line
        self.set_draw_color(*self.colors['navy'])
        self.set_line_width(0.8)
        self.line(50, 105, 160, 105)
        
        # Report metadata box with enhanced design
        self.set_xy(30, 130)
        self.set_fill_color(*self.colors['light_text'])
        self.set_draw_color(*self.colors['navy'])
        self.set_line_width(0.8)
        self.rect(30, 130, 150, 100, 'DF')
        
        # Metadata header with pink accent
        self.set_fill_color(*self.colors['pink'])
        self.set_text_color(*self.colors['light_text'])
        self.set_font('Arial', 'B', 16)
        self.set_xy(30, 130)
        self.cell(150, 12, "REPORT SUMMARY", 0, 1, 'C', True)
        
        # Metadata content
        self.set_font('Arial', '', 11)
        self.set_text_color(*self.colors['dark_text'])
        
        info_lines = [
            ("Report ID:", f"{self.report_id}"),
            ("Generated:", f"{self.creation_time}"),
            ("Patient Name:", f"{patient_info.get('patient_name', 'N/A')}"),
            ("Age:", f"{patient_info.get('age', 'N/A')}"),
            ("Gender:", f"{patient_info.get('gender', 'N/A').title()}"),
            ("Generated by:", "BreastCareAI")
        ]
        
        for i, (label, value) in enumerate(info_lines):
            y_position = 150 + i * 8
            # Label in navy
            self.set_text_color(*self.colors['navy'])
            self.set_font('Arial', 'B', 10)
            self.set_xy(40, y_position)
            self.cell(50, 6, label, 0, 0, 'L')
            # Value in dark text
            self.set_text_color(*self.colors['dark_text'])
            self.set_font('Arial', '', 10)
            self.set_xy(90, y_position)
            self.cell(80, 6, value, 0, 1, 'L')
        
        # Confidential notice
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*self.colors['pink'])
        self.set_xy(0, 250)
        self.cell(0, 6, "FOR EDUCATIONAL AND INFORMATIONAL USE ONLY", 0, 1, 'C')
        
        self.ln(20)
    
    def create_section_title(self, title, fill_bg=True):
        """Create a styled section title with breast cancer awareness colors"""
        if fill_bg:
            self.set_fill_color(*self.colors['light_blue'])
        else:
            self.set_fill_color(*self.colors['light_text'])
        
        self.set_text_color(*self.colors['navy'])
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L', fill_bg)
        self.ln(3)
    
    def create_patient_info_table(self, patient_info):
        """Create a clean, properly aligned patient information table"""
        self.create_section_title("PATIENT INFORMATION")
        
        # Table setup
        col_width = 180 / 2
        row_height = 8
        
        # Table header
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(*self.colors['navy'])
        self.set_text_color(*self.colors['light_text'])
        self.cell(col_width, row_height, "FIELD", 1, 0, 'C', True)
        self.cell(col_width, row_height, "INFORMATION", 1, 1, 'C', True)
        
        # Patient data
        patient_data = {
            "Patient Name": patient_info.get('patient_name', 'Not provided'),
            "Age": patient_info.get('age', 'Not provided'),
            "Gender": patient_info.get('gender', 'Not provided').title(),
            "Medical History": patient_info.get('medical_history', 'Not provided') or 'Not provided'
        }
        
        # Table rows with alternating colors
        self.set_font('Arial', '', 10)
        
        for i, (field, value) in enumerate(patient_data.items()):
            # Alternating row colors
            if i % 2 == 0:
                self.set_fill_color(*self.colors['light_blue'])
            else:
                self.set_fill_color(*self.colors['light_pink'])
            
            # Field column
            self.set_text_color(*self.colors['navy'])
            self.set_font('Arial', 'B', 10)
            self.cell(col_width, row_height, field, 1, 0, 'L', True)
            
            # Value column
            self.set_text_color(*self.colors['dark_text'])
            self.set_font('Arial', '', 10)
            
            # Handle long text with wrapping
            sanitized_value = sanitize_text_for_pdf(str(value))
            if len(sanitized_value) > 40:
                # Store current position
                x = self.get_x()
                y = self.get_y()
                
                # Multi-cell for long values
                self.multi_cell(col_width, 4, sanitized_value, 1, 'L', True)
                # Reset position for next row
                self.set_xy(x + col_width, y + max(8, (len(sanitized_value) // 40) * 4))
            else:
                self.cell(col_width, row_height, sanitized_value, 1, 1, 'L', True)
        
        self.ln(5)
    
    def create_prediction_summary(self, prediction_data, overall_confidence):
        """Create prediction summary with color-coded results"""
        self.create_section_title("REPORT SUMMARY")
        
        # Determine result color and styling
        is_malignant = prediction_data.get("prediction") == "Breast Cancer Detected"
        malignant_prob = float(prediction_data.get('malignant', 0.0)) * 100
        benign_prob = float(prediction_data.get('benign', 0.0)) * 100
        
        if is_malignant:
            result_color = self.colors['danger']
            result_bg = (254, 226, 226)  # Light red background
            status_text = "HIGH RISK - IMMEDIATE MEDICAL ATTENTION RECOMMENDED"
        else:
            result_color = self.colors['success']
            result_bg = (220, 252, 231)  # Light green background
            status_text = "LOW RISK - CONTINUE REGULAR SCREENING"
        
        # Main result box
        self.set_fill_color(*result_bg)
        self.set_text_color(*result_color)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 12, prediction_data.get("prediction", "N/A"), 0, 1, 'C', True)
        
        # Status text
        self.set_font('Arial', 'I', 10)
        self.set_text_color(*self.colors['medium_text'])
        self.cell(0, 6, status_text, 0, 1, 'C')
        
        self.ln(5)
        
        # Confidence metrics in a clean table
        summary_data = {
            "Overall Confidence": f"{max(malignant_prob, benign_prob):.1f}%",
            "Primary Model": prediction_data.get("model", "N/A"),
            "Malignant Probability": f"{malignant_prob:.1f}%",
            "Benign Probability": f"{benign_prob:.1f}%",
            "Analysis Method": "Multi-Model AI Consensus",
            "Generated by": "BreastCareAI"
        }
        
        # Display summary with proper spacing
        self.set_font('Arial', '', 10)
        
        for i, (key, value) in enumerate(summary_data.items()):
            # Alternating background
            if i % 2 == 0:
                self.set_fill_color(*self.colors['light_blue'])
            else:
                self.set_fill_color(*self.colors['light_pink'])
            
            # Key with professional styling
            self.set_text_color(*self.colors['navy'])
            self.set_font('Arial', 'B', 10)
            self.cell(70, 8, f" {key}", 1, 0, 'L', True)
            
            # Value with color coding
            if "Malignant" in key and float(value.replace('%', '')) > 50:
                self.set_text_color(*self.colors['danger'])
            elif "Benign" in key and float(value.replace('%', '')) > 50:
                self.set_text_color(*self.colors['success'])
            else:
                self.set_text_color(*self.colors['dark_text'])
                
            self.set_font('Arial', '', 10)
            self.cell(110, 8, value, 1, 1, 'L', True)
        
        self.ln(8)
    
    def create_model_predictions_table(self, results):
        """Create detailed model predictions table with clean alignment"""
        self.create_section_title("DETAILED MODEL ANALYSIS")
        
        # Check if we need a new page for the table
        if self.get_y() > 220:
            self.add_page()
        
        # Table setup with optimized column widths
        col_widths = [50, 35, 20, 20, 25, 30]  # Total: 180
        headers = ["AI Model", "Prediction", "Benign %", "Malignant %", "Confidence %", "Accuracy %"]
        
        # Table header with brand colors
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(*self.colors['navy'])
        self.set_text_color(*self.colors['light_text'])
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln(8)
        
        # Table rows
        self.set_font('Arial', '', 8)
        
        for i, (key, item) in enumerate(results.items()):
            # Check for page break
            if self.get_y() > 270:
                self.add_page()
                # Reprint header on new page
                self.set_font('Arial', 'B', 9)
                self.set_fill_color(*self.colors['navy'])
                self.set_text_color(*self.colors['light_text'])
                for j, header in enumerate(headers):
                    self.cell(col_widths[j], 8, header, 1, 0, 'C', True)
                self.ln(8)
                self.set_font('Arial', '', 8)
            
            # Alternating row colors
            if i % 2 == 0:
                self.set_fill_color(*self.colors['light_blue'])
            else:
                self.set_fill_color(*self.colors['light_pink'])
            
            # Prepare data
            model_name = sanitize_text_for_pdf(item.get("display", key))
            pred = item.get("predicted_label", "N/A")
            p_benign = float(item.get("proba", {}).get("benign", 0.0)) * 100.0
            p_malign = float(item.get("proba", {}).get("malignant", 0.0)) * 100.0
            confidence = float(item.get("confidence", 0.0)) * 100.0
            
            # Handle accuracy
            acc_val = item.get("accuracy", None)
            if acc_val is None:
                default_accuracies = {
                    "svm": 0.95, "random_forest": 0.96, "logistic_regression": 0.93
                }
                acc_val = default_accuracies.get(key, 0.90)
            acc_str = f"{acc_val * 100:.1f}" if isinstance(acc_val, (int, float)) else "N/A"
            
            # Model name
            self.set_text_color(*self.colors['navy'])
            self.set_font('Arial', 'B', 8)
            self.cell(col_widths[0], 7, model_name, 1, 0, 'L', True)
            
            # Prediction with color coding
            if pred.lower().startswith("malignant"):
                pred_color = self.colors['danger']
            else:
                pred_color = self.colors['success']
            
            self.set_text_color(*pred_color)
            self.set_font('Arial', 'B', 8)
            self.cell(col_widths[1], 7, pred, 1, 0, 'C', True)
            
            # Probabilities
            self.set_font('Arial', '', 8)
            
            # Benign probability
            self.set_text_color(*self.colors['success'])
            self.cell(col_widths[2], 7, f"{p_benign:.1f}", 1, 0, 'C', True)
            
            # Malignant probability
            self.set_text_color(*self.colors['danger'])
            self.cell(col_widths[3], 7, f"{p_malign:.1f}", 1, 0, 'C', True)
            
            # Confidence
            self.set_text_color(*self.colors['soft_blue'])
            self.cell(col_widths[4], 7, f"{confidence:.1f}", 1, 0, 'C', True)
            
            # Accuracy
            self.set_text_color(*self.colors['medium_text'])
            self.cell(col_widths[5], 7, acc_str, 1, 1, 'C', True)
        
        self.ln(8)
    
    def create_recommendations_section(self, recommendations):
        """Create recommendations section with clean formatting"""
        self.create_section_title("MEDICAL RECOMMENDATIONS & NEXT STEPS")
        
        self.set_font('Arial', '', 10)
        self.set_text_color(*self.colors['dark_text'])
        
        for i, rec in enumerate(recommendations):
            if self.get_y() > 270:
                self.add_page()
            
            if rec.startswith("-"):
                # Bullet point - use hyphen instead of bullet to avoid encoding issues
                self.set_font('Arial', '', 9)
                self.set_text_color(*self.colors['navy'])
                self.cell(5)
                self.cell(5, 5, "-", 0, 0, 'L')
                self.set_text_color(*self.colors['dark_text'])
                self.multi_cell(0, 5, sanitize_text_for_pdf(rec[2:]))
            elif rec.endswith(":"):
                # Sub-header
                self.set_font('Arial', 'B', 10)
                self.set_text_color(*self.colors['pink'])
                self.multi_cell(0, 6, sanitize_text_for_pdf(rec))
                self.set_text_color(*self.colors['dark_text'])
            elif rec == "":
                # Empty line for spacing
                self.ln(4)
            elif rec.startswith("IMMEDIATE") or "HIGH RISK" in rec:
                # Critical recommendation
                self.set_font('Arial', 'B', 10)
                self.set_text_color(*self.colors['danger'])
                self.multi_cell(0, 6, sanitize_text_for_pdf(rec))
                self.set_text_color(*self.colors['dark_text'])
            elif "MODERATE" in rec or "LOW RISK" in rec:
                # Moderate recommendation
                self.set_font('Arial', 'B', 10)
                self.set_text_color(*self.colors['warning'])
                self.multi_cell(0, 6, sanitize_text_for_pdf(rec))
                self.set_text_color(*self.colors['dark_text'])
            else:
                # Regular text
                self.set_font('Arial', '', 9)
                self.multi_cell(0, 5, sanitize_text_for_pdf(rec))
            
            self.ln(2)
    
    def create_disclaimer_section(self):
        """Create medical disclaimer section with required note"""
        if self.get_y() > 180:  # Ensure enough space
            self.add_page()
        
        self.create_section_title("IMPORTANT MEDICAL DISCLAIMER")
        
        # Disclaimer box with light background
        self.set_fill_color(*self.colors['light_gray'])
        self.set_draw_color(*self.colors['navy'])
        self.set_line_width(0.5)
        disclaimer_height = 65
        self.rect(self.left_margin, self.get_y(), 180, disclaimer_height, 'DF')
        
        # Disclaimer text
        self.set_xy(self.left_margin + 5, self.get_y() + 5)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*self.colors['danger'])
        self.cell(0, 5, "MEDICAL DISCLAIMER & LIMITATIONS", 0, 1, 'L')
        
        self.ln(2)
        
        # Main disclaimer text
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*self.colors['dark_text'])
        
        disclaimer_text = (
            "This AI-generated report is for informational purposes only and should NOT be considered as a definitive "
            "medical diagnosis. The predictions presented herein are based on machine learning models trained on historical "
            "data and have inherent limitations. This analysis does not replace professional medical evaluation, diagnosis, "
            "or treatment by qualified healthcare providers. Always consult with certified medical professionals for proper "
            "medical diagnosis, treatment decisions, and follow-up care. BreastCareAI and its developers expressly disclaim "
            "any liability for medical decisions made based on this report."
        )
        
        # Multi-cell with justified alignment
        self.multi_cell(170, 4, sanitize_text_for_pdf(disclaimer_text), 0, 'J')
        
        self.ln(8)
        
        # Required note - clearly visible below disclaimer box
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*self.colors['pink'])
        self.cell(0, 5, "Note: Regular screening and clinical evaluation are essential for accurate breast health assessment.", 0, 1, 'L')
        
        self.ln(45)  # Space after disclaimer

# ========= Routes =========
@app.route("/")
def redirect_to_welcome():
    return redirect("/welcome")

@app.route("/welcome")
def welcome():
    logger.debug("Rendering welcome page")
    # Get user data if logged in
    user_data = None
    if "user_id" in session:
        user_data = get_user_by_id(session["user_id"])
    return render_template("welcome.html", user=user_data)

@app.route("/common-page")
@login_required
def common_page():
    """Common page where users choose prediction type"""
    user_data = get_user_by_id(session["user_id"])
    return render_template("common_page.html", user=user_data)

@app.route("/data-prediction")
@login_required
def data_prediction():
    """Data-based prediction input page"""

    # Read first (do not destroy immediately)
    form_data = session.get("form_data", {})
    error_message = session.get("error_message")

    # Clear after reading (safe cleanup)
    session.pop("form_data", None)
    session.pop("error_message", None)
    session.modified = True

    user_data = get_user_by_id(session["user_id"])

    return render_template(
        "data_prediction.html",
        user=user_data,
        form_data=form_data,
        error_message=error_message
    )


@app.route("/profile")
@login_required
def profile():
    """Enhanced profile page with user history and statistics"""
    user_data = get_user_by_id(session["user_id"])
    
    # Get user history and statistics
    user_history = get_user_history(session["user_id"])
    user_stats = get_user_stats(session["user_id"])
    
    return render_template(
        "profile.html", 
        user=user_data, 
        user_history=user_history,
        user_stats=user_stats
    )

@app.route("/edit-profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    user_data = get_user_by_id(session["user_id"])
    
    if request.method == "GET":
        # Pass today's date for date validation
        today = datetime.now().strftime('%Y-%m-%d')
        return render_template("edit_profile.html", user=user_data, today=today)
    
    # POST request handling
    if request.method == "POST":
        # Check if it's an AJAX request
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        # Get form data
        name = request.form.get('name', '').strip()
        date_of_birth = request.form.get('date_of_birth', '').strip()
        gender = request.form.get('gender', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        # ========== SERVER-SIDE VALIDATION ==========
        errors = []
        
        # 1. Validate Full Name (MANDATORY)
        if not name:
            errors.append("Full name is required.")
        elif len(name) < 3:
            errors.append("Full name must be at least 3 characters long.")
        elif not re.match(r'^[A-Za-z\s]{3,}$', name):
            errors.append("Full name can only contain letters and spaces.")
        
        # 2. Validate Phone Number (OPTIONAL but strict if provided)
        if phone:  # Only validate if phone is provided
            # Remove any spaces, dashes, or parentheses
            clean_phone = re.sub(r'[\s\-\(\)]+', '', phone)
            
            # Check if only digits remain
            if not clean_phone.isdigit():
                errors.append("Phone number must contain only digits.")
            elif len(clean_phone) < 10 or len(clean_phone) > 15:
                errors.append("Phone number must be 10-15 digits long.")
            else:
                # Update phone with cleaned version
                phone = clean_phone
        
        # 3. Validate Date of Birth (OPTIONAL)
        if date_of_birth:
            # Validate YYYY-MM-DD format
            try:
                birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
                # Ensure date is not in the future
                if birth_date > datetime.now():
                    errors.append("Date of birth cannot be in the future.")
            except ValueError:
                errors.append("Invalid date format. Please use YYYY-MM-DD format.")
        
        # 4. Validate Gender (OPTIONAL)
        if gender:
            valid_genders = ['female', 'male', 'other', 'prefer_not_to_say']
            if gender not in valid_genders:
                errors.append("Invalid gender selection.")
        
        # ========== HANDLE VALIDATION ERRORS ==========
        if errors:
            # Join errors into a single, readable message
            error_message = " ".join(errors)
            
            if is_ajax:
                # AJAX response: JSON with 400 status
                return jsonify({
                    "success": False,
                    "message": error_message
                }), 400
            else:
                # Non-AJAX: Flash messages and re-render
                for error in errors:
                    flash(error, "danger")
                today = datetime.now().strftime('%Y-%m-%d')
                return render_template("edit_profile.html", user=user_data, today=today)
        
        # ========== ALL VALIDATIONS PASSED - UPDATE DATABASE ==========
        try:
            # Update user profile in database
            success = update_user_profile(
                session["user_id"], 
                name, 
                date_of_birth if date_of_birth else None, 
                gender if gender else None, 
                phone if phone else None, 
                address if address else None
            )
            
            if success:
                # Update session with new name for consistency
                session["user_name"] = name
                
                if is_ajax:
                    # AJAX response: Simple success JSON with 200 status
                    return jsonify({"success": True}), 200
                else:
                    # Non-AJAX: Flash success and redirect
                    flash("Profile updated successfully!", "success")
                    return redirect(url_for('profile'))
            else:
                # Database update failed
                error_msg = "Failed to update profile. Please try again."
                
                if is_ajax:
                    return jsonify({
                        "success": False,
                        "message": error_msg
                    }), 500
                else:
                    flash(error_msg, "danger")
                    
        except Exception as e:
            # Log the actual error for debugging
            logger.error(f"Profile update error: {str(e)}")
            
            # User-friendly error message
            error_msg = "An internal error occurred while updating your profile."
            
            if is_ajax:
                return jsonify({
                    "success": False,
                    "message": error_msg
                }), 500
            else:
                flash(error_msg, "danger")
        
        # ========== NON-AJAX FALLBACK FOR ERRORS ==========
        # This handles non-AJAX requests when update fails
        if not is_ajax:
            today = datetime.now().strftime('%Y-%m-%d')
            return render_template("edit_profile.html", user=user_data, today=today)
    
    # This should not be reached in normal flow
    # Only for invalid request methods or edge cases
    if is_ajax:
        return jsonify({
            "success": False,
            "message": "Invalid request"
        }), 400
    else:
        return redirect(url_for('edit_profile'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect("/common-page")

    if request.method == "POST":
        try:
            # Sanitize inputs
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            date_of_birth = request.form.get('date_of_birth')
            gender = request.form.get('gender')

            # -------- BASIC REQUIRED CHECK --------
            if not all([name, email, password, confirm_password]):
                return jsonify({'success': False, 'field': 'form', 'message': 'All required fields must be filled.'})

            # -------- NAME VALIDATION --------
            if not re.match(r'^[A-Za-z\s]{3,50}$', name):
                return jsonify({'success': False, 'field': 'name', 'message': 'Name must contain only letters and spaces (min 3 characters).'})

            # -------- EMAIL VALIDATION --------
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return jsonify({'success': False, 'field': 'email', 'message': 'Invalid email address format.'})

            # -------- PASSWORD MATCH --------
            if password != confirm_password:
                return jsonify({'success': False, 'field': 'confirm_password', 'message': 'Passwords do not match.'})

            # -------- STRONG PASSWORD POLICY --------
            if len(password) < 8:
                return jsonify({'success': False, 'field': 'password', 'message': 'Password must be at least 8 characters long.'})

            if not re.search(r'[A-Z]', password):
                return jsonify({'success': False, 'field': 'password', 'message': 'Password must include an uppercase letter.'})

            if not re.search(r'[a-z]', password):
                return jsonify({'success': False, 'field': 'password', 'message': 'Password must include a lowercase letter.'})

            if not re.search(r'\d', password):
                return jsonify({'success': False, 'field': 'password', 'message': 'Password must include a number.'})

            if not re.search(r'[^\w\s]', password):
                return jsonify({'success': False, 'field': 'password', 'message': 'Password must include a special character.'})

            # -------- DATE OF BIRTH VALIDATION --------
            if date_of_birth:
                try:
                    dob = datetime.strptime(date_of_birth, '%Y-%m-%d')
                    today = datetime.today()
                    age = (today - dob).days // 365

                    if dob > today:
                        return jsonify({'success': False, 'field': 'date_of_birth', 'message': 'Date of birth cannot be in the future.'})

                    if age < 13 or age > 120:
                        return jsonify({'success': False, 'field': 'date_of_birth', 'message': 'Age must be between 13 and 120.'})

                except ValueError:
                    return jsonify({'success': False, 'field': 'date_of_birth', 'message': 'Invalid date format.'})

            # -------- GENDER VALIDATION --------
            if gender and gender not in ['male', 'female', 'other']:
                return jsonify({'success': False, 'field': 'gender', 'message': 'Invalid gender selection.'})

            # -------- REGISTER USER --------
            register_user(name, email, password, date_of_birth, gender)

            session['registration_success'] = True
            session['registered_user_name'] = name

            return jsonify({
                'success': True,
                'message': 'Account created successfully.',
                'redirect_url': url_for('login')
            })

        except Exception as e:
            if "already exists" in str(e).lower():
                return jsonify({'success': False, 'field': 'email', 'message': 'Email is already registered.'})

            return jsonify({'success': False, 'message': 'Registration failed. Please try again.'})

    return render_template("register.html", form_data={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect("/common-page")

    # Registration success messages
    registration_success = session.pop('registration_success', False)
    registered_user_name = session.pop('registered_user_name', None)

    if request.method == "POST":
        try:
            email = request.form['email'].lower().strip()
            password = request.form['password']
            remember = request.form.get('remember') == 'yes'

            if not email or not password:
                flash("Please fill in all fields.", "danger")
                return render_template("login.html", form_data=request.form)

            user = validate_login(email, password)
            if user:
                # 🔐 Clear old session safely
                session.clear()

                # ✅ Store user data
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]

                # ✅ Remember-me logic (NO global lifetime change)
                session.permanent = remember

                session.modified = True

                flash(f"Welcome back, {user['name']}!", "success")

                # Respect next parameter if present
                next_page = request.args.get("next")
                return redirect(next_page or "/welcome")

            else:
                flash("Invalid email or password.", "danger")
                return render_template("login.html", form_data=request.form)

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash("An error occurred during login. Please try again.", "danger")
            return render_template("login.html", form_data=request.form)

    return render_template(
        "login.html",
        form_data={},
        registration_success=registration_success,
        registered_user_name=registered_user_name
    )


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    """Handle logout with AJAX support"""
    user_name = session.get("user_name")
    session.clear()
    
    # Return JSON response for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'message': f'You have been logged out. Goodbye, {user_name}!'
        })
    
    # For non-AJAX requests
    flash(f"You have been logged out. Goodbye, {user_name}!", "info")
    return redirect("/welcome")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        try:
            email = request.form["email"].lower().strip()
            user = check_email_exists(email)
            if user:
                # Store email in session for reset password page
                session['reset_email'] = email
                
                # Return JSON response for AJAX handling
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'message': 'Please reset your password.',
                        'redirect_url': url_for('reset_password')
                    })
                else:
                    flash("Please reset your password.", "info")
                    return redirect("/reset-password")
            else:
                # Don't reveal whether email exists for security
                message = "If an account with that email exists, we have sent a password reset link."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'message': message
                    })
                else:
                    flash(message, "success")
                    return render_template("forgot_password.html")
        except Exception as e:
            logger.error(f"Forgot password error: {str(e)}")
            error_message = "An error occurred. Please try again."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'message': error_message
                })
            else:
                flash(error_message, "danger")
                return render_template("forgot_password.html")
    
    return render_template("forgot_password.html", form_data={})

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    # Get email from session or query parameter
    email = session.get('reset_email') or request.args.get('email', '')
    
    if request.method == "POST":
        try:
            email = request.form["email"].lower().strip()
            new_password = request.form["new_password"]
            confirm_password = request.form["confirm_password"]
            
            # Validation
            if new_password != confirm_password:
                error_message = "Passwords do not match."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
            if len(new_password) < 8:
                error_message = "Password must be at least 8 characters long."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
            # Enhanced password validation
            if not any(char.isupper() for char in new_password):
                error_message = "Password must contain at least one uppercase letter."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
            if not any(char.islower() for char in new_password):
                error_message = "Password must contain at least one lowercase letter."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
            if not any(char.isdigit() for char in new_password):
                error_message = "Password must contain at least one number."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
            if not any(not char.isalnum() for char in new_password):
                error_message = "Password must contain at least one special character."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
            
            # Reset password
            if reset_user_password(email, new_password):
                # Clear reset email from session
                session.pop('reset_email', None)
                
                # Store success message in session for login page
                session['password_reset_success'] = True
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'message': 'Password reset successfully! Please log in with your new password.',
                        'redirect_url': url_for('login')
                    })
                else:
                    flash("Password reset successfully! Please log in with your new password.", "success")
                    return redirect("/login")
            else:
                error_message = "Password reset failed. Please try again."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'message': error_message
                    })
                else:
                    flash(error_message, "danger")
                    return render_template("reset_password.html", email=email)
                
        except Exception as e:
            logger.error(f"Reset password error: {str(e)}")
            error_message = "An error occurred during password reset. Please try again."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'message': error_message
                })
            else:
                flash(error_message, "danger")
                return render_template("reset_password.html", email=request.form.get("email", ""))
    
    # Check if we have a valid email for reset
    if not email:
        flash("Invalid or expired password reset link. Please try again.", "danger")
        return redirect("/forgot-password")
    
    return render_template("reset_password.html", email=email)

@app.route("/predict", methods=["POST"])
@login_required
@models_loaded_required
def predict():
    try:
        logger.info("=== PREDICTION REQUEST STARTED ===")
        logger.info(f"User: {session.get('user_name')}")

        # ---------------- VALIDATION ----------------
        patient_name = request.form.get('patient_name', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        medical_history = request.form.get('medical_history', '').strip()

        errors = []
        if not patient_name:
            errors.append("Patient name is required")
        elif not re.match(r'^[A-Za-z\s]+$', patient_name):
            errors.append("Patient name must contain only letters and spaces")

        if not age:
            errors.append("Age is required")
        else:
            try:
                age_val = float(age)
                if not 0 <= age_val <= 120:
                    errors.append("Age must be between 0 and 120")
            except ValueError:
                errors.append("Age must be a valid number")

        if not gender or gender not in ['female', 'male', 'other']:
            errors.append("Invalid gender selection")

        if errors:
            error_message = "; ".join(errors)
            flash(f"Input error: {error_message}", "danger")
            session["form_data"] = request.form.to_dict()
            session["error_message"] = error_message
            session.modified = True
            return redirect(url_for('data_prediction'))

        # ---------------- PATIENT INFO ----------------
        patient_info = {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "medical_history": medical_history
        }
        session["patient_info"] = patient_info

        # ---------------- PREDICTION ----------------
        selected_models = get_selected_models_from_form(request.form)
        feature_row = build_feature_row_from_form(request.form)
        X_scaled = scaler.transform(feature_row)

        results, model_selection_text = make_results_for_single_sample(
            X_scaled, selected_models
        )

        benign_confidence, malignant_confidence = calculate_overall_confidence(results)
        risk_level = malignant_confidence
        is_malignant = malignant_confidence > benign_confidence

        recommendations = get_recommendations(risk_level, is_malignant)

        # ---------------- SESSION (CRITICAL FIX) ----------------
        # ✅ Stable master keys for result + PDF
        session["prediction_results"] = results
        session["best_prediction"] = pick_best_for_pdf(results)
        session["overall_confidence"] = {
            "benign": benign_confidence,
            "malignant": malignant_confidence
        }
        session["model_selection_text"] = model_selection_text

        # Keep backward compatibility (do NOT remove)
        session["last_results"] = results
        session["last_prediction"] = session["best_prediction"]

        session.modified = True  # 🔥 VERY IMPORTANT

        # ---------------- HISTORY ----------------
        for model_key, result_data in results.items():
            add_user_history(
                user_id=session["user_id"],
                prediction_type="data_prediction",
                model_used=result_data["display"],
                result=result_data["predicted_label"],
                confidence=result_data["confidence"],
                input_data={
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender
                }
            )

        # ---------------- UI DATA ----------------
        max_malignant_prob = max(
            item["proba"]["malignant"] for item in results.values()
        )
        precautions = get_precautions(max_malignant_prob)

        patient_features = {
            FEATURE_NAMES[i]: feature_row[0][i]
            for i in range(len(FEATURE_NAMES))
        }

        user_data = get_user_by_id(session["user_id"])

        return render_template(
            "data_result.html",
            user=user_data,
            payload={
                "results": results,
                "model_selection_text": model_selection_text
            },
            precautions=precautions,
            feature_names=FEATURE_NAMES,
            feature_values=feature_row.flatten().tolist(),
            patient_info=patient_info,
            patient_features=patient_features,
            benign_averages=BENIGN_AVERAGES,
            malignant_averages=MALIGNANT_AVERAGES,
            feature_importance=FEATURE_IMPORTANCE,
            overall_confidence=session["overall_confidence"],
            risk_level=risk_level,
            is_malignant=is_malignant,
            recommendations=recommendations,
            form_data=request.form.to_dict()
        )

    except ValueError as e:
        logger.warning("Input validation failed: %s", str(e))
        flash(f"Input error: {str(e)}", "danger")
        session["form_data"] = request.form.to_dict()
        session["error_message"] = str(e)
        session.modified = True
        return redirect(url_for('data_prediction'))

    except Exception as e:
        logger.error("Prediction error occurred", exc_info=True)
        flash(
            "An error occurred during prediction. Please try again or contact support.",
            "danger"
        )
        session["form_data"] = request.form.to_dict()
        session.modified = True
        return redirect(url_for('data_prediction'))


@app.route("/download-report")
@login_required
def download_report():
    """Enhanced PDF download with BreastCareAI branding and breast cancer awareness theme"""
    try:
        # ---------------- SESSION DATA CHECK ----------------
        results = session.get("prediction_results") or session.get("last_results")
        last_prediction = session.get("best_prediction") or session.get("last_prediction")
        patient_info = session.get("patient_info")
        model_selection_text = session.get(
            "model_selection_text", "Unknown Models Selected"
        )

        if not results or not last_prediction or not patient_info:
            logger.warning("No prediction data found for PDF generation")
            flash("No prediction data available. Please run a prediction first.", "danger")
            return redirect(url_for('data_prediction'))

        # ---------------- USER DATA ----------------
        user_data = get_user_by_id(session["user_id"])
        if not user_data:
            logger.error("User data not found for PDF generation")
            flash("User data not found. Please log in again.", "danger")
            return redirect(url_for('login'))

        logger.info(
            f"Generating BreastCareAI PDF report for user: {user_data['name']}"
        )

        # ---------------- CONFIDENCE ----------------
        overall_confidence = session.get(
            "overall_confidence",
            calculate_overall_confidence(results)
        )

        # ---------------- RECOMMENDATIONS ----------------
        is_malignant = last_prediction.get("prediction") == "Breast Cancer Detected"
        malignant_prob = float(last_prediction.get("malignant", 0.0))

        recommendations = get_detailed_recommendations(
            last_prediction.get("prediction"),
            malignant_prob,
            patient_info.get("age"),
            patient_info.get("gender")
        )

        # ---------------- PDF GENERATION ----------------
        pdf = BreastCareAIPDFReport()

        # Cover page
        pdf.create_cover_page(user_data, patient_info)

        # Content pages
        pdf.add_page()
        pdf.create_patient_info_table(patient_info)
        pdf.create_prediction_summary(last_prediction, overall_confidence)
        pdf.create_model_predictions_table(results)
        pdf.create_recommendations_section(recommendations)
        pdf.create_disclaimer_section()

        # ---------------- OUTPUT ----------------
        pdf_output = pdf.output(dest="S")

        # Safe encoding
        pdf_data = pdf_output.encode("latin-1", "replace")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BreastCareAI_Report_{timestamp}.pdf"

        logger.info("BreastCareAI PDF report generated successfully")

        return send_file(
            io.BytesIO(pdf_data),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error("Download report error", exc_info=True)
        flash("An error occurred while generating the report. Please try again.", "danger")
        return redirect(url_for('data_prediction'))



@app.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    try:
        user_id = session.get("user_id")
        user_name = session.get("user_name")

        if not user_id:
            # Session expired or invalid
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "message": "Session expired"}), 401
            flash("Session expired. Please log in again.", "warning")
            return redirect("/login")

        # Perform deletion
        deletion_success = delete_user_account(user_id)

        if deletion_success:
            logger.info(f"Account deleted successfully for user: {user_name} (ID: {user_id})")

            # Clear session safely
            session.clear()

            # ✅ AJAX request → return JSON (for toast)
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({
                    "success": True,
                    "message": "Your account has been deleted successfully."
                }), 200

            # ✅ Normal form submit fallback
            flash("Your account has been deleted successfully.", "info")
            return redirect("/welcome")

        else:
            logger.warning(f"Account deletion failed for user ID: {user_id}")

            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({
                    "success": False,
                    "message": "Account deletion failed. Please try again."
                }), 400

            flash("Account deletion failed. Please try again.", "danger")
            return redirect("/profile")

    except Exception as e:
        logger.error("Account deletion error", exc_info=True)

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "success": False,
                "message": "An unexpected error occurred while deleting your account."
            }), 500

        flash("An error occurred while deleting your account. Please try again.", "danger")
        return redirect("/profile")


# ========= New Routes for Additional Features =========
@app.route("/image-prediction", methods=["GET", "POST"])
@login_required
def image_prediction():
    """Image-based breast cancer prediction using CNN"""
    
    user_data = get_user_by_id(session["user_id"])

    if request.method == "POST":
        try:
            # 1️⃣ Check file presence
            if "image" not in request.files:
                flash("Please select an image file.", "danger")
                return render_template("image_prediction.html", user=user_data)

            image_file = request.files["image"]

            if image_file.filename == "":
                flash("Please select an image file.", "danger")
                return render_template("image_prediction.html", user=user_data)

            # 2️⃣ Validate file extension (ONLY PNG)
            if not image_file.filename.lower().endswith(".png"):
                flash("Only PNG images are supported for this prediction.", "danger")
                return render_template("image_prediction.html", user=user_data)

            # 3️⃣ Preprocess image for CNN
            image_array = preprocess_image(image_file)

            # 4️⃣ CNN Prediction
            malignant_prob = predict_image(image_array)
            benign_prob = 1 - malignant_prob

            # 5️⃣ Determine result
            if malignant_prob >= 0.5:
                result = "Malignant"
                confidence = malignant_prob
                color = "danger"
            else:
                result = "Benign"
                confidence = benign_prob
                color = "success"

            # 6️⃣ Convert image to base64 (for preview)
            image_file.seek(0)
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # 7️⃣ Recommendations & precautions
            recommendations = get_recommendations(malignant_prob, result == "Malignant")
            precautions = get_precautions(malignant_prob)

            # 8️⃣ Save prediction history
            add_user_history(
                user_id=session["user_id"],
                prediction_type="image_prediction",
                model_used="CNN (Histopathology Images)",
                result=result,
                confidence=confidence,
                input_data={
                    "filename": image_file.filename
                }
            )

            # 9️⃣ Render result
            return render_template(
                "image_prediction.html",
                user=user_data,
                prediction_made=True,
                result=result,
                confidence=round(confidence * 100, 2),
                malignant_prob=round(malignant_prob * 100, 2),
                benign_prob=round(benign_prob * 100, 2),
                color=color,
                image_data=image_data,
                recommendations=recommendations,
                precautions=precautions
            )

        except Exception as e:
            logger.error(f"Image prediction error: {str(e)}", exc_info=True)
            flash("Error processing image. Please try again.", "danger")
            return render_template("image_prediction.html", user=user_data)

    # GET request
    return render_template("image_prediction.html", user=user_data)


@app.route("/symptoms-assessment", methods=["GET", "POST"])
@login_required
def symptoms_assessment():
    """Symptoms assessment page with enhanced validation and safety checks"""
    user_data = get_user_by_id(session["user_id"])
    
    if request.method == "POST":
        try:
            answers = {}
            missing_questions = []
            
            # 1️⃣ Validation Safety: Ensure all questions are answered server-side
            for question in SYMPTOMS_QUESTIONS:
                question_id = question['id']
                question_text = question['question']
                
                # Get answer from form
                if question['type'] == 'yes_no':
                    answer = request.form.get(question_id, '').strip().lower()
                    # Validate yes_no answers
                    if answer not in ['yes', 'no']:
                        missing_questions.append({
                            'id': question_id,
                            'question': question_text,
                            'index': len(answers) + 1
                        })
                        answers[question_id] = ''  # Mark as empty
                    else:
                        answers[question_id] = answer
                else:
                    answer = request.form.get(question_id, '').strip()
                    if not answer:  # Empty answer
                        missing_questions.append({
                            'id': question_id,
                            'question': question_text,
                            'index': len(answers) + 1
                        })
                        answers[question_id] = ''
                    else:
                        answers[question_id] = answer
            
            # If any answers are missing, re-render with error
            if missing_questions:
                error_message = f"Please answer all {len(SYMPTOMS_QUESTIONS)} questions before submitting."
                if len(missing_questions) == 1:
                    error_message = f"Please answer question {missing_questions[0]['index']}: '{missing_questions[0]['question']}'"
                elif len(missing_questions) <= 3:
                    missing_nums = ', '.join(str(q['index']) for q in missing_questions)
                    error_message = f"Please answer questions {missing_nums} before submitting."
                
                flash(error_message, "warning")
                return render_template(
                    "symptoms_assessment.html",
                    user=user_data,
                    questions=SYMPTOMS_QUESTIONS,
                    validation_error=True,
                    missing_questions=[q['index'] for q in missing_questions],
                    answers=answers,  # Include partial answers for better UX
                    assessment_completed=False
                )
            
            # 2️⃣ Defensive Coding: Handle unexpected input safely
            # Verify we have answers for all expected questions
            expected_ids = {q['id'] for q in SYMPTOMS_QUESTIONS}
            received_ids = set(answers.keys())
            
            if expected_ids != received_ids:
                logger.warning(f"Question ID mismatch. Expected: {expected_ids}, Received: {received_ids}")
                # Fill missing answers with defaults
                for qid in expected_ids - received_ids:
                    answers[qid] = 'no' if next((q for q in SYMPTOMS_QUESTIONS if q['id'] == qid and q['type'] == 'yes_no'), False) else ''
            
            # Calculate risk score (existing logic)
            risk_result = calculate_symptoms_risk(answers)
            
            # Validate risk calculation result
            if not risk_result or 'risk_category' not in risk_result:
                logger.error(f"Invalid risk calculation result: {risk_result}")
                raise ValueError("Risk calculation failed")
            
            # 3️⃣ Robust Result Context: Ensure all required data is available
            # Get recommendations based on risk
            is_high_risk = risk_result.get('risk_category') == "High Risk"
            risk_percentage = risk_result.get('risk_percentage', 0)
            recommendations = get_recommendations(risk_percentage / 100, is_high_risk)
            
            # Ensure recommendations is always a list
            if not isinstance(recommendations, list):
                recommendations = [
                    "Consult with a healthcare provider for personalized advice.",
                    "Consider scheduling regular breast cancer screenings based on your age and risk factors.",
                    "Maintain a healthy lifestyle with regular exercise and balanced nutrition."
                ]
            
            # Ensure risk_factors exists in risk_result
            if 'risk_factors' not in risk_result:
                risk_result['risk_factors'] = []
            
            # Ensure color exists for risk category
            if 'color' not in risk_result:
                if risk_result['risk_category'] == "High Risk":
                    risk_result['color'] = '#ef4444'  # red
                elif risk_result['risk_category'] == "Moderate Risk":
                    risk_result['color'] = '#f59e0b'  # amber
                else:
                    risk_result['color'] = '#10b981'  # emerald
            
            # Record user history
            try:
                add_user_history(
                    user_id=session["user_id"],
                    prediction_type="symptoms_assessment",
                    model_used="Symptoms Assessment Algorithm",
                    result=risk_result['risk_category'],
                    confidence=risk_percentage / 100,
                    input_data=answers
                )
            except Exception as history_error:
                logger.error(f"Failed to save user history: {str(history_error)}")
                # Continue even if history fails - don't break the user experience
            
            return render_template(
                "symptoms_assessment.html",
                user=user_data,
                assessment_completed=True,
                answers=answers,
                risk_result=risk_result,
                recommendations=recommendations,
                questions=SYMPTOMS_QUESTIONS
            )
            
        except Exception as e:
            logger.error(f"Symptoms assessment error: {str(e)}", exc_info=True)
            flash("An unexpected error occurred during assessment. Please try again.", "danger")
            
            # Return to assessment with all necessary context
            return render_template(
                "symptoms_assessment.html",
                user=user_data,
                questions=SYMPTOMS_QUESTIONS,
                assessment_completed=False,
                validation_error=True
            )
    
    # GET request - initial page load
    return render_template(
        "symptoms_assessment.html",
        user=user_data,
        questions=SYMPTOMS_QUESTIONS,
        assessment_completed=False
    )

# ========= Error Handlers =========
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {str(error)}", exc_info=True)
    return render_template('500.html'), 500

# ========= Security Headers =========
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    if os.environ.get('FLASK_ENV') == 'production':
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.after_request  
def add_no_cache_headers(response):
    if request.path.startswith('/static/'):
        return response
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.context_processor
def inject_user():
    """Automatically inject user info into all templates."""
    user_data = None
    if "user_id" in session:
        user_data = get_user_by_id(session["user_id"])
    return dict(user=user_data)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)