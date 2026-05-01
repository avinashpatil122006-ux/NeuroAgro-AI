import os
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify
# Import utilities
from NeuroPredict.utils.predict import load_models
from NeuroPredict.utils.preprocess import preprocess_crop_input
app = Flask(__name__)

# Load all ML models upon starting the application
load_models()

@app.route('/')
def home():
    """Render the main dashboard dashboard."""
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def crop_api():
    try:
        data = request.form
        features = preprocess_crop_input(data)
        result = predict_crop(features)
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_irrigation', methods=['POST'])
def irrigation_api():
    try:
        data = request.form
        features = preprocess_irrigation_input(data)
        result = predict_irrigation(features)
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_disease', methods=['POST'])
def disease_api():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded.'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file.'})
            
        image_bytes = file.read()
        features = preprocess_disease_image(image_bytes)
        result, confidence = predict_disease(features)
        
        return jsonify({
            'success': True, 
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_yield', methods=['POST'])
def yield_api():
    try:
        data = request.form
        features = preprocess_yield_input(data)
        result = predict_yield(features)
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def open_browser():
    """Automatically launch Google Chrome (or default browser) to the app url."""
    url = "http://127.0.0.1:5000/"
    try:
        # Try to explicitly open Chrome if available
        webbrowser.get('chrome').open(url)
    except:
        # Fallback to default browser
        webbrowser.open(url)

if __name__ == '__main__':
    # Add a timer to open the browser shortly after the server starts
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True, use_reloader=False)
