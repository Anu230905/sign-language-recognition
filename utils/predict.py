"""
utils/predict.py
-----------------
Shared helper functions for loading the trained model and running
predictions on either an uploaded image or a webcam frame.

Model expects: 28x28 GRAYSCALE images (Sign Language MNIST format).
"""

import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 28


def load_sign_model(model_path="model/sign_language_model.h5",
                     class_map_path="model/class_indices.json"):
    model = load_model(model_path)
    with open(class_map_path, "r") as f:
        idx_to_class = json.load(f)
    # JSON keys come back as strings; convert back to int
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class


def preprocess_frame(frame_bgr):
    """Takes a BGR (OpenCV) frame/image and returns a model-ready tensor."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    tensor = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return tensor


def predict_frame(model, idx_to_class, frame_bgr):
    """Returns (predicted_label, confidence, full_probability_array)."""
    tensor = preprocess_frame(frame_bgr)
    preds = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    label = idx_to_class[best_idx]
    confidence = float(preds[best_idx])
    return label, confidence, preds
