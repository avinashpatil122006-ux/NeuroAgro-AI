import numpy as np
import io
from PIL import Image

def preprocess_crop_input(data):
    """
    Preprocess the crop recommendation inputs.
    Expects data dictionary with keys: N, P, K, temperature, humidity, ph, rainfall.
    """
    try:
        features = [
            float(data.get('N', 0)),
            float(data.get('P', 0)),
            float(data.get('K', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0))
        ]
        # Reshape to (1, 7) for sklearn
        return np.array(features).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Invalid input for crop recommendation: {str(e)}")

def preprocess_irrigation_input(data):
    """
    Preprocess the irrigation prediction inputs.
    Expects data dictionary with keys: soil_moisture, temperature, humidity.
    """
    try:
        features = [
            float(data.get('soil_moisture', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0))
        ]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Invalid input for irrigation prediction: {str(e)}")

def preprocess_disease_image(image_bytes):
    """
    Preprocess the image for CNN model.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize to 128x128 as expected by our CNN
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0 # normalize
        # Reshape to (1, 128, 128, 3) for tensorflow
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Invalid image format or processing error: {str(e)}")

def preprocess_yield_input(data):
    """
    Preprocess the yield prediction inputs.
    Expects data dictionary with keys: year, rainfall, pesticides, avg_temp.
    """
    try:
        features = [
            float(data.get('year', 0)),
            float(data.get('rainfall', 0)),
            float(data.get('pesticides', 0)),
            float(data.get('avg_temp', 0))
        ]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Invalid input for yield prediction: {str(e)}")
