import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import cv2
import base64
from io import BytesIO
import sys

app = Flask(__name__)
import sys
import os

model1_path = 'models/model1_normal_vs_amd.pkl'
model1 = joblib.load(model1_path)

model2_path = 'models/model2_cnv_vs_drusen.pkl'
model2 = joblib.load(model2_path)

def extract_features(image):
    """Extracts column-wise white pixel distances as features."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply binary thresholding
    _, binary_thresh = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

    # Calculate distances between top-most and bottom-most white pixels per column
    distances = []
    for col in range(binary_thresh.shape[1]):
        top_pixel, bottom_pixel = -1, -1

        # Find top-most white pixel
        for row in range(binary_thresh.shape[0]):
            if binary_thresh[row, col] == 255:
                top_pixel = row
                break

        # Find bottom-most white pixel
        for row in range(binary_thresh.shape[0] - 1, -1, -1):
            if binary_thresh[row, col] == 255:
                bottom_pixel = row
                break

        # Calculate distance
        distance = abs(bottom_pixel - top_pixel) if top_pixel != -1 and bottom_pixel != -1 else 0
        distances.append(distance)

    return np.array(distances).reshape(1, -1)  # Reshape for model input

def mark_bottom_points(image, binary_thresh):
    """Marks the bottom-most white pixel for each column with a red dot."""
    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for red dots

    for col in range(binary_thresh.shape[1]):
        # Find the bottom-most white pixel
        for row in range(binary_thresh.shape[0] - 1, -1, -1):
            if binary_thresh[row, col] == 255:
                # Mark with a red dot (BGR format: Red is (0, 0, 255))
                cv2.circle(marked_image, (col, row), 5, (0, 0, 255), -1)
                break

    return marked_image

@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict AMD, DRUSEN, or CNV based on uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Convert uploaded image to grayscale
    image_data = file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize image to 496x496
    image = cv2.resize(image, (496, 496))

    features = extract_features(image)

    # Thresholding to get the binary thresholded image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_thresh = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

    # Encode the binary thresholded image to base64
    _, encoded_binary_thresh = cv2.imencode('.jpg', binary_thresh)
    base64_binary_thresh = base64.b64encode(encoded_binary_thresh).decode("utf-8")

    # Mark the bottom-most white pixels on the thresholded image
    marked_image = mark_bottom_points(image, binary_thresh)
    _, encoded_marked_image = cv2.imencode('.jpg', marked_image)
    base64_marked_image = base64.b64encode(encoded_marked_image).decode("utf-8")

    # Encode the original resized image to base64
    _, encoded_image = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(encoded_image).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"

    # Model 1 prediction
    prob1 = model1.predict_proba(features)[0]
    labels1 = model1.classes_
    pred1 = labels1[np.argmax(prob1)]
    normal_vs_amd_probs = {labels1[i]: prob1[i] for i in range(len(labels1))}

    # Model 2 prediction (only if not NORMAL)
    prob2 = model2.predict_proba(features)[0]
    labels2 = model2.classes_
    pred2 = labels2[np.argmax(prob2)]
    drusen_vs_cnv_probs = {labels2[i]: prob2[i] for i in range(len(labels2))}

    return jsonify({
        "prediction": pred1,
        "probabilities": normal_vs_amd_probs,
        "normal_vs_amd_probabilities": normal_vs_amd_probs,
        "image": image_url,
        "binary_thresholded_image": f"data:image/jpeg;base64,{base64_binary_thresh}",
        "marked_image": f"data:image/jpeg;base64,{base64_marked_image}",
        "cnv_vs_drusen_probabilities": drusen_vs_cnv_probs if pred1 != "NORMAL" else {},
        "second_model_prediction": pred2 if pred1 != "NORMAL" else "Predicted as NORMAL"
    })

import webbrowser
import threading

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.25, open_browser).start()
    app.run()

