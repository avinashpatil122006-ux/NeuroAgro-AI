import os
import joblib
import numpy as np

# Global variables
CROP_MODEL = None
IRRIGATION_MODEL = None
YIELD_MODEL = None

def load_models(base_path='NeuroPredict/models'):
    global CROP_MODEL, IRRIGATION_MODEL, YIELD_MODEL

    # Crop model
    crop_path = os.path.join(base_path, 'crop_model.pkl')
    if os.path.exists(crop_path):
        CROP_MODEL = joblib.load(crop_path)

    # Irrigation model
    irrig_path = os.path.join(base_path, 'irrigation_model.pkl')
    if os.path.exists(irrig_path):
        IRRIGATION_MODEL = joblib.load(irrig_path)

    # Yield model
    yield_path = os.path.join(base_path, 'yield_model.pkl')
    if os.path.exists(yield_path):
        YIELD_MODEL = joblib.load(yield_path)


def predict_crop(features):
    if CROP_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = CROP_MODEL.predict(features)
    return prediction[0]


def predict_irrigation(features):
    if IRRIGATION_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = IRRIGATION_MODEL.predict(features)
    return "Yes, irrigation needed." if prediction[0] == 1 else "No irrigation needed."


def predict_yield(features):
    if YIELD_MODEL is None:
        return "Model not found. Run train_models.py first."
    prediction = YIELD_MODEL.predict(features)
    return f"{prediction[0]:.2f} Tons / Hectare"
