import os
import joblib
import tensorflow as tf
import numpy as np

# Global variables to hold models
CROP_MODEL = None
IRRIGATION_MODEL = None
YIELD_MODEL = None
DISEASE_MODEL = None

DISEASE_CLASSES = ['Healthy', 'Powdery Mildew', 'Rust', 'Blight']

def load_models(base_path='models'):
    global CROP_MODEL, IRRIGATION_MODEL, YIELD_MODEL, DISEASE_MODEL
    
    # Load Crop Model
    crop_path = os.path.join(base_path, 'crop_model.pkl')
    if os.path.exists(crop_path):
        CROP_MODEL = joblib.load(crop_path)
        
    # Load Irrigation Model
    irrig_path = os.path.join(base_path, 'irrigation_model.pkl')
    if os.path.exists(irrig_path):
        IRRIGATION_MODEL = joblib.load(irrig_path)
        
    # Load Yield Model
    yield_path = os.path.join(base_path, 'yield_model.pkl')
    if os.path.exists(yield_path):
        YIELD_MODEL = joblib.load(yield_path)
        
    # Load Disease Model
    disease_path = os.path.join(base_path, 'disease_model.h5')
    if os.path.exists(disease_path):
        DISEASE_MODEL = tf.keras.models.load_model(disease_path)

def predict_crop(features):
    if CROP_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = CROP_MODEL.predict(features)
    return prediction[0]

def predict_irrigation(features):
    if IRRIGATION_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = IRRIGATION_MODEL.predict(features)
    return "Yes, irrigation needed." if prediction[0] == 1 else "No, sufficient moisture."

def predict_yield(features):
    if YIELD_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = YIELD_MODEL.predict(features)
    return f"{prediction[0]:.2f} Tons / Hectare"

def predict_disease(image_array):
    if DISEASE_MODEL is None:
        return "Model not found. Run train_models.py first.", 0.0
    
    prediction = DISEASE_MODEL.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    return DISEASE_CLASSES[class_idx], float(confidence) * 100
