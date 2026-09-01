import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from flask import Flask, request, jsonify, render_template


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "best_xgboost.joblib"
CLASSES_FILE = "gesture_classes.joblib"
FEATURES_FILE = "feature_columns.joblib"


model = joblib.load(MODEL_FILE)
gesture_classes = joblib.load(CLASSES_FILE)
feature_columns = joblib.load(FEATURES_FILE)


print("Model loaded successfully")
print("Number of classes:", len(gesture_classes))


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_hand_landmarks(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    landmarks = {}

    for i, landmark in enumerate(hand.landmark):

        # Dataset uses x1...x21
        landmarks[f"x{i + 1}"] = landmark.x
        landmarks[f"y{i + 1}"] = landmark.y

    return pd.DataFrame([landmarks])


# ============================================================
# NORMALIZE COORDINATES
# ============================================================

def normalize_coordinates(df):

    x_coords = [
        col for col in df.columns
        if col.startswith("x")
    ]

    y_coords = [
        col for col in df.columns
        if col.startswith("y")
    ]

    x_vals = df[x_coords].copy()
    y_vals = df[y_coords].copy()

    wrist_x = x_vals.iloc[:, 0].copy()
    wrist_y = y_vals.iloc[:, 0].copy()

    # Move wrist to origin
    x_vals = x_vals.subtract(
        wrist_x,
        axis=0
    )

    y_vals = y_vals.subtract(
        wrist_y,
        axis=0
    )

    # Same normalization used during training
    distance = np.sqrt(
        x_vals["x13"] ** 2 +
        y_vals["y13"] ** 2
    )

    distance = distance.replace(
        0,
        1e-8
    )

    x_vals = x_vals.div(
        distance,
        axis=0
    )

    y_vals = y_vals.div(
        distance,
        axis=0
    )

    normalized_data = {}

    for x_col, y_col in zip(
        x_vals.columns,
        y_vals.columns
    ):

        normalized_data[x_col] = x_vals[x_col]
        normalized_data[y_col] = y_vals[y_col]

    return pd.DataFrame(
        normalized_data,
        index=df.index
    )


# ============================================================
# FINGER DISTANCE FEATURES
# ============================================================

def calculate_finger_tip_distances(data):

    data = data.copy()

    TIP_INDICES = {
        "thumb": 5,
        "index": 9,
        "middle": 13,
        "ring": 17,
        "pinky": 21
    }

    FINGER_BASES = {
        "thumb": 2,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18
    }

    # Pairwise fingertip distances
    tips = list(TIP_INDICES.items())

    for i in range(len(tips)):

        name1, tip1 = tips[i]

        for j in range(i + 1, len(tips)):

            name2, tip2 = tips[j]

            data[f"dist_{name1}_{name2}"] = np.sqrt(
                (data[f"x{tip1}"] - data[f"x{tip2}"]) ** 2
                +
                (data[f"y{tip1}"] - data[f"y{tip2}"]) ** 2
            )

    # Thumb distances
    thumb_x = data["x5"]
    thumb_y = data["y5"]

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        tip = TIP_INDICES[finger]

        data[f"dist_thumb_{finger}"] = np.sqrt(
            (thumb_x - data[f"x{tip}"]) ** 2
            +
            (thumb_y - data[f"y{tip}"]) ** 2
        )

    # Finger lengths
    for finger, base in FINGER_BASES.items():

        tip = TIP_INDICES[finger]

        data[f"len_{finger}"] = np.sqrt(
            (data[f"x{base}"] - data[f"x{tip}"]) ** 2
            +
            (data[f"y{base}"] - data[f"y{tip}"]) ** 2
        )

    return data


# ============================================================
# PREDICT
# ============================================================

def predict_gesture(image):

    landmarks = extract_hand_landmarks(
        image
    )

    if landmarks is None:
        return "No Hand Detected", 0.0

    # Normalize
    normalized = normalize_coordinates(
        landmarks
    )

    # Add engineered features
    features = calculate_finger_tip_distances(
        normalized
    )

    # IMPORTANT:
    # Ensure exact same column order as training
    features = features.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    probabilities = model.predict_proba(
        features
    )[0]

    best_index = int(
        np.argmax(probabilities)
    )

    gesture = gesture_classes[
        best_index
    ]

    confidence = float(
        probabilities[best_index]
    )

    return gesture, confidence


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image received"
        }), 400

    file = request.files["image"]

    image_bytes = file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return jsonify({
            "error": "Invalid image"
        }), 400

    gesture, confidence = predict_gesture(
        image
    )

    return jsonify({
        "gesture": gesture,
        "confidence": round(
            confidence * 100,
            2
        )
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
