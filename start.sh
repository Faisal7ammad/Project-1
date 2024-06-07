#!/bin/bash

# Start TensorFlow Serving
tensorflow_model_server --rest_api_port=8501 --model_name=project1 --model_base_path=/models/project1 &

# Start Flask app
cd /app
gunicorn app:app --bind 0.0.0.0:5000 --workers 1
