import pickle
import os
import numpy as np
import logging

# Set up logger
logger = logging.getLogger(__name__)

def load_models():
    """
    Load all models and scaler from the 'model' directory
    This function aligns with the model loading in app.py
    """
    base_path = os.path.join(os.getcwd(), 'model')
    
    models = {}
    scaler = None
    
    try:
        # Define model paths (same as in app.py)
        model_paths = {
            'svm': os.path.join(base_path, 'svm_model.pkl'),
            'random_forest': os.path.join(base_path, 'random_forest_model.pkl'),
            'logistic_regression': os.path.join(base_path, 'logistic_regression_model.pkl')
        }
        
        # Load scaler first
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Successfully loaded scaler")
        
        # Load each model
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                continue
                
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            logger.info(f"Successfully loaded model: {model_name}")
        
        if not models:
            raise Exception("No models were successfully loaded")
            
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise Exception(f"Model file not found: {e}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise Exception(f"Error loading models: {e}")

    return models, scaler

def predict_from_models(models, data, model_names):
    """
    Predict using selected models
    Enhanced to handle different model types and provide better error handling
    """
    results = {}
    
    # If no specific models selected, use all available models
    if not model_names or 'all' in model_names:
        model_names = list(models.keys())
    
    for name in model_names:
        model = models.get(name)
        if not model:
            results[name] = {'error': f"Model '{name}' not found."}
            continue
            
        try:
            # Make prediction
            prediction = model.predict(data)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data)
                # Handle different probability formats
                if probabilities.shape[1] == 2:
                    proba_benign = float(probabilities[0][0])
                    proba_malignant = float(probabilities[0][1])
                else:
                    # For binary classification with single probability output
                    proba_malignant = float(probabilities[0][0])
                    proba_benign = 1.0 - proba_malignant
            else:
                # Fallback for models without predict_proba
                pred_value = int(prediction[0])
                proba_malignant = float(pred_value)
                proba_benign = 1.0 - proba_malignant
            
            # Determine label based on probability
            predicted_label = "Malignant" if proba_malignant >= 0.5 else "Benign"
            confidence = max(proba_benign, proba_malignant)
            
            results[name] = {
                'prediction': int(prediction[0]),
                'predicted_label': predicted_label,
                'probability': {
                    'benign': proba_benign,
                    'malignant': proba_malignant
                },
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with model {name}: {str(e)}")
            results[name] = {
                'error': f"Prediction failed: {str(e)}",
                'success': False
            }

    return results

def ensure_proba(model, X):
    """
    Ensure probability output for any model type
    Fallback to decision_function or predict if predict_proba not available
    """
    try:
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
        
        # Fallback: use predict and create binary probabilities
        preds = model.predict(X)
        out = np.zeros((len(preds), 2), dtype=float)
        for i, y in enumerate(preds):
            if int(y) == 1:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
            else:
                out[i, 0] = 1.0
                out[i, 1] = 0.0
        return out
        
    except Exception as e:
        logger.error(f"Error ensuring probabilities: {str(e)}")
        # Ultimate fallback
        preds = model.predict(X)
        out = np.zeros((len(preds), 2), dtype=float)
        for i, y in enumerate(preds):
            if int(y) == 1:
                out[i, 1] = 1.0
            else:
                out[i, 0] = 1.0
        return out

def map_proba_to_benign_malignant(model, proba_row):
    """
    Map probability row to benign/malignant consistently
    """
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