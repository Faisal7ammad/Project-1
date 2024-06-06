from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8501/v1/models/project1:predict")


def preprocess_image(image):
    # Resize the image to the expected input shape
    image = image.resize((32, 32))
    image = np.array(image).astype(np.float32)
    # Normalize the image
    mean, variance = tf.nn.moments(image, axes=[0, 1, 2])
    image_normalized = (image - mean.numpy()) / tf.sqrt(variance).numpy()
    return image_normalized

def decode_label(label):
    # Replace with actual label decoding logic
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return labels[label]

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the image file
        image = Image.open(file.stream)
        image_preprocessed = preprocess_image(image)
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

        # Prepare the payload for TensorFlow Serving
        payload = {
            "instances": image_preprocessed.tolist()
        }

        # Make the request to TensorFlow Serving
        response = requests.post(MODEL_URL, json=payload)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.content}")

        if response.status_code != 200:
            return jsonify({'error': 'Request to TensorFlow Serving failed', 'details': response.content.decode()}), response.status_code

        try:
            predictions = response.json()['predictions']
        except KeyError as e:
            return jsonify({'error': 'KeyError', 'details': str(e), 'response': response.json()})

        # Convert predictions to readable format
        results = np.argmax(predictions, axis=1)[0]
        decoded_label = decode_label(results)

        return jsonify({'prediction': decoded_label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=os.getenv("PORT", default=5000))
