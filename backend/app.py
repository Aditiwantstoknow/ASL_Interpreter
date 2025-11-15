from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import json
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
MODEL_PATH = '../model/isl_model.h5'
LABELS_PATH = '../model/labels.json'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to accept prediction

# Load model and labels
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully!")

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)
print(f"✓ Loaded {len(class_labels)} classes")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def decode_base64_image(base64_string):
    """Decode base64 image from frontend"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB (in case it's RGBA)
    img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert RGB to BGR (OpenCV format)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'num_classes': len(class_labels)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sign language letter from image"""
    try:
        # Get image from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Preprocess
        processed_img = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get predicted label
        predicted_label = class_labels[str(predicted_class_idx)]
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': True,
                'letter': '?',
                'confidence': confidence,
                'below_threshold': True,
                'message': 'Low confidence - could not recognize clearly'
            })
        
        # Return prediction
        return jsonify({
            'success': True,
            'letter': predicted_label,
            'confidence': confidence,
            'below_threshold': False,
            'all_predictions': {
                class_labels[str(i)]: float(predictions[0][i]) 
                for i in range(len(class_labels))
            }
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available classes"""
    return jsonify({
        'classes': list(class_labels.values()),
        'num_classes': len(class_labels)
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ISL INTERPRETER BACKEND SERVER")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {len(class_labels)}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%")
    print("="*50)
    print("\n🚀 Server starting on http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)