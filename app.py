import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from flask import Flask, render_template, request, jsonify


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


# ============================================================
# Load trained model and metadata
# ============================================================

MODEL_FILE = "best_xgboost.joblib"
CLASSES_FILE = "gesture_classes.joblib"
FEATURES_FILE = "feature_columns.joblib"


if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. "
        "Run train_model.py first."
    )

if not os.path.exists(CLASSES_FILE):
    raise FileNotFoundError(
        f"{CLASSES_FILE} not found."
    )

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(
        f"{FEATURES_FILE} not found."
    )


model = joblib.load(MODEL_FILE)
gesture_classes = joblib.load(CLASSES_FILE)
feature_columns = joblib.load(FEATURES_FILE)


print("Model loaded successfully")
print("Number of classes:", len(gesture_classes))
print("Number of features:", len(feature_columns))


# ============================================================
# MediaPipe
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ============================================================
# Extract hand landmarks
# ============================================================

def extract_hand_landmarks(image):
    """
    Extract 21 MediaPipe hand landmarks.

    Returns:
        DataFrame containing x1,y1,...,x21,y21
        or None if no hand is detected.
    """

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = {}

    for i, landmark in enumerate(
        hand_landmarks.landmark,
        start=1
    ):
        landmarks[f"x{i}"] = landmark.x
        landmarks[f"y{i}"] = landmark.y

    return pd.DataFrame([landmarks])


# ============================================================
# Normalize coordinates
# ============================================================

def normalize_coordinates(df):
    """
    Normalize hand landmarks exactly like training pipeline.
    """

    df = df.copy()

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

    # Wrist = landmark 1
    wrist_x = x_vals["x1"]
    wrist_y = y_vals["y1"]

    # Recenter around wrist
    x_vals = x_vals.subtract(
        wrist_x,
        axis=0
    )

    y_vals = y_vals.subtract(
        wrist_y,
        axis=0
    )

    # Same scale reference used during training
    # landmark 13 = middle finger tip in your dataset
    scale = np.sqrt(
        x_vals["x13"] ** 2 +
        y_vals["y13"] ** 2
    )

    scale = scale.replace(0, 1)

    x_vals = x_vals.div(
        scale,
        axis=0
    )

    y_vals = y_vals.div(
        scale,
        axis=0
    )

    # Interleave x,y
    normalized_data = {}

    for i in range(1, 22):

        normalized_data[f"x{i}"] = x_vals[f"x{i}"]

        normalized_data[f"y{i}"] = y_vals[f"y{i}"]

    normalized_df = pd.DataFrame(
        normalized_data,
        index=df.index
    )

    return normalized_df


# ============================================================
# Finger distance features
# ============================================================

def calculate_finger_tip_distances(df):
    """
    Create the same additional features used during training.
    """

    df = df.copy()

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

    # --------------------------------------------------------
    # Pairwise finger tip distances
    # --------------------------------------------------------

    tips = list(TIP_INDICES.items())

    for i in range(len(tips)):

        name1, tip1 = tips[i]

        for j in range(i + 1, len(tips)):

            name2, tip2 = tips[j]

            df[f"dist_{name1}_{name2}"] = np.sqrt(
                (df[f"x{tip1}"] - df[f"x{tip2}"]) ** 2
                +
                (df[f"y{tip1}"] - df[f"y{tip2}"]) ** 2
            )

    # --------------------------------------------------------
    # Thumb to other finger tips
    # --------------------------------------------------------

    thumb_x = df["x5"]
    thumb_y = df["y5"]

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        tip = TIP_INDICES[finger]

        df[f"dist_thumb_{finger}"] = np.sqrt(
            (thumb_x - df[f"x{tip}"]) ** 2
            +
            (thumb_y - df[f"y{tip}"]) ** 2
        )

    # --------------------------------------------------------
    # Finger lengths
    # --------------------------------------------------------

    for finger, base in FINGER_BASES.items():

        tip = TIP_INDICES[finger]

        df[f"len_{finger}"] = np.sqrt(
            (df[f"x{base}"] - df[f"x{tip}"]) ** 2
            +
            (df[f"y{base}"] - df[f"y{tip}"]) ** 2
        )

    return df


# ============================================================
# Complete preprocessing
# ============================================================

def preprocess_image(image):
    """
    Image -> MediaPipe landmarks -> normalized features
    """

    landmarks = extract_hand_landmarks(image)

    if landmarks is None:
        return None

    normalized = normalize_coordinates(
        landmarks
    )

    enhanced = calculate_finger_tip_distances(
        normalized
    )

    # --------------------------------------------------------
    # Ensure exact training feature order
    # --------------------------------------------------------

    for column in feature_columns:

        if column not in enhanced.columns:
            enhanced[column] = 0.0

    enhanced = enhanced[
        feature_columns
    ]

    enhanced = enhanced.astype(
        np.float32
    )

    return enhanced


# ============================================================
# Prediction
# ============================================================

def predict_gesture(image):

    features = preprocess_image(
        image
    )

    if features is None:
        return {
            "gesture": "No Hand Detected",
            "confidence": 0.0
        }

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

    return {
        "gesture": str(gesture),
        "confidence": confidence
    }


# ============================================================
# Home page
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Health check
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "classes": len(gesture_classes),
        "features": len(feature_columns)
    })


# ============================================================
# Prediction API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
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

        result = predict_gesture(
            image
        )

        return jsonify(result)

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Render entry point
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
