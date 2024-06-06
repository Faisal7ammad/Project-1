# Base image for TensorFlow Serving
FROM tensorflow/serving:latest as serving

# Install necessary packages
FROM python:3.9-slim

# Install Flask and other dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy the saved model to the serving directory
COPY project-1-model_4_p3_s14_saved_model /models/project1/1

# Copy Flask app files
COPY app.py /app/app.py
COPY templates /app/templates

# Set the working directory
WORKDIR /app

# Expose the Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
