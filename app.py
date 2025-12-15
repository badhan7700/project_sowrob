"""
Traffic Sign Recognition Web Application
Flask backend for GTSRB classification using Simple CNN and MobileNetV2
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import base64

app = Flask(__name__)
CORS(app)

# Traffic sign class names (GTSRB 43 classes)
SIGN_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Model cache
models = {}

def load_models():
    """Load both models at startup"""
    global models
    try:
        if os.path.exists('simple_cnn_final.h5'):
            models['simple_cnn'] = load_model('simple_cnn_final.h5')
            print("Simple CNN loaded successfully")
    except Exception as e:
        print(f"Error loading Simple CNN: {e}")
    
    try:
        if os.path.exists('mobilenetv2_final.h5'):
            # Model uses Rescaling layer (no Lambda), loads directly
            models['mobilenetv2'] = load_model('mobilenetv2_final.h5')
            print("MobileNetV2 loaded successfully")
    except Exception as e:
        print(f"Error loading MobileNetV2: {e}")
        import traceback
        traceback.print_exc()


def preprocess_image(image, model_type):
    """Preprocess image based on model type"""
    if model_type == 'simple_cnn':
        # Resize to 32x32 and normalize to [0,1]
        img = image.resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)
    else:  # mobilenetv2
        # Resize to 96x96 and normalize to [0,1]
        img = image.resize((96, 96))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 96, 96, 3)
    return img_array


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
        
        # Get model type
        model_type = request.form.get('model', 'simple_cnn')
        if model_type not in models:
            return jsonify({'error': f'Model {model_type} not available'}), 400
        
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(image, model_type)
        
        # Run prediction
        model = models[model_type]
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        results = []
        for idx in top_indices:
            results.append({
                'class_id': int(idx),
                'class_name': SIGN_NAMES[idx],
                'confidence': float(predictions[idx] * 100)
            })
        
        return jsonify({
            'success': True,
            'model_used': 'Simple CNN' if model_type == 'simple_cnn' else 'MobileNetV2',
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Return available models"""
    available = []
    if 'simple_cnn' in models:
        available.append({'id': 'simple_cnn', 'name': 'Simple CNN (Fast, Lightweight)'})
    if 'mobilenetv2' in models:
        available.append({'id': 'mobilenetv2', 'name': 'MobileNetV2 (More Accurate)'})
    return jsonify({'models': available})


if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print(f"Available models: {list(models.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5000)
