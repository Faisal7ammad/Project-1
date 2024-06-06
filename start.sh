#!/bin/bash

# Start TensorFlow Serving
tensorflow_model_server --rest_api_port=8501 --model_name=${MODEL_NAME} 
--model_base_path=/models/${MODEL_NAME} &

# Start Flask app
python3 /app/app.py
