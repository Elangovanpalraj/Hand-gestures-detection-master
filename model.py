# ============================================================
# model.py
# Hand Gesture Detection - XGBoost Backend
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import cv2
import mediapipe as mp

from flask import Flask, request, jsonify

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "hand_landmarks_data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "best_xgboost.joblib"
)

CLASSES_FILE = os.path.join(
    BASE_DIR,
    "gesture_classes.joblib"
)

FEATURES_FILE = os.path.join(
    BASE_DIR,
    "feature_columns.joblib"
)

# Gesture classes
GESTURE_CLASSES = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted"
]


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL MODEL
# ============================================================

model = None
feature_columns = None
gesture_classes = GESTURE_CLASSES


# ============================================================
# CHECK FILE
# ============================================================

def check_data_file():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"\nERROR: Dataset not found:\n{DATA_FILE}\n\n"
            "Make sure hand_landmarks_data.csv is uploaded to GitHub."
        )


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_coordinates(df):

    df = df.copy()

    # Get x and y columns
    x_coords = [
        col for col in df.columns
        if str(col).lower().startswith("x")
    ]

    y_coords = [
        col for col in df.columns
        if str(col).lower().startswith("y")
    ]

    # Sort numerically
    x_coords = sorted(
        x_coords,
        key=lambda x: int(x[1:])
    )

    y_coords = sorted(
        y_coords,
        key=lambda x: int(x[1:])
    )

    if len(x_coords) == 0 or len(y_coords) == 0:

        raise ValueError(
            "X/Y landmark columns were not found."
        )

    x_vals = df[x_coords].copy()
    y_vals = df[y_coords].copy()

    # Wrist
    wrist_x = x_vals.iloc[:, 0].copy()
    wrist_y = y_vals.iloc[:, 0].copy()

    # Recenter
    x_vals = x_vals.subtract(
        wrist_x,
        axis=0
    )

    y_vals = y_vals.subtract(
        wrist_y,
        axis=0
    )

    # ========================================================
    # Scale normalization
    # Dataset uses x13/y13
    # ========================================================

    if "x13" in x_vals.columns and "y13" in y_vals.columns:

        scale = np.sqrt(
            x_vals["x13"] ** 2 +
            y_vals["y13"] ** 2
        )

    else:

        # fallback: distance wrist -> last landmark
        last_x = x_vals.iloc[:, -1]
        last_y = y_vals.iloc[:, -1]

        scale = np.sqrt(
            last_x ** 2 +
            last_y ** 2
        )

    # Avoid divide by zero
    scale = scale.replace(
        0,
        1e-8
    )

    x_vals = x_vals.div(
        scale,
        axis=0
    )

    y_vals = y_vals.div(
        scale,
        axis=0
    )

    # ========================================================
    # Interleave X and Y
    # x1,y1,x2,y2,...
    # ========================================================

    normalized_data = {}

    for x_col, y_col in zip(
        x_vals.columns,
        y_vals.columns
    ):

        normalized_data[x_col] = x_vals[x_col]
        normalized_data[y_col] = y_vals[y_col]

    normalized_df = pd.DataFrame(
        normalized_data,
        index=df.index
    )

    return normalized_df


# ============================================================
# FINGER DISTANCE FEATURES
# ============================================================

def calculate_finger_tip_distances(df):

    df = df.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # These indices match your training code.
    # --------------------------------------------------------

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

    # ========================================================
    # Check required columns
    # ========================================================

    required = []

    for idx in TIP_INDICES.values():
        required.extend([
            f"x{idx}",
            f"y{idx}"
        ])

    for idx in FINGER_BASES.values():
        required.extend([
            f"x{idx}",
            f"y{idx}"
        ])

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing landmark columns: {missing}"
        )

    # ========================================================
    # Pairwise finger-tip distances
    # ========================================================

    tips = list(TIP_INDICES.items())

    for i in range(len(tips)):

        name1, tip1 = tips[i]

        for j in range(i + 1, len(tips)):

            name2, tip2 = tips[j]

            dist = np.sqrt(
                (
                    df[f"x{tip1}"] -
                    df[f"x{tip2}"]
                ) ** 2
                +
                (
                    df[f"y{tip1}"] -
                    df[f"y{tip2}"]
                ) ** 2
            )

            df[
                f"dist_{name1}_{name2}"
            ] = dist

    # ========================================================
    # Thumb-to-other distances
    # ========================================================

    thumb_x = df["x5"]
    thumb_y = df["y5"]

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        tip = TIP_INDICES[finger]

        df[
            f"dist_thumb_{finger}"
        ] = np.sqrt(
            (
                thumb_x -
                df[f"x{tip}"]
            ) ** 2
            +
            (
                thumb_y -
                df[f"y{tip}"]
            ) ** 2
        )

    # ========================================================
    # Finger lengths
    # ========================================================

    for finger, base in FINGER_BASES.items():

        tip = TIP_INDICES[finger]

        df[
            f"len_{finger}"
        ] = np.sqrt(
            (
                df[f"x{base}"] -
                df[f"x{tip}"]
            ) ** 2
            +
            (
                df[f"y{base}"] -
                df[f"y{tip}"]
            ) ** 2
        )

    return df


# ============================================================
# CREATE FEATURES FROM RAW CSV
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove Z columns
    # --------------------------------------------------------

    z_columns = [
        col
        for col in df.columns
        if "z" in str(col).lower()
    ]

    df_2d = df.drop(
        columns=z_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if "label" not in df_2d.columns:

        raise ValueError(
            "Dataset must contain a 'label' column."
        )

    labels = df_2d["label"].copy()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = normalize_coordinates(
        df_2d.drop(columns=["label"])
    )

    # --------------------------------------------------------
    # Add distance features
    # --------------------------------------------------------

    enhanced = calculate_finger_tip_distances(
        normalized
    )

    # --------------------------------------------------------
    # Add label
    # --------------------------------------------------------

    enhanced["label"] = labels.values

    return enhanced


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global model
    global feature_columns
    global gesture_classes

    print("=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    check_data_file()

    print(
        f"Loading dataset: {DATA_FILE}"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Original dataset shape:",
        df.shape
    )

    # ========================================================
    # Feature engineering
    # ========================================================

    final_df = create_features(
        df
    )

    print(
        "Final feature shape:",
        final_df.shape
    )

    # ========================================================
    # X / y
    # ========================================================

    X = final_df.drop(
        columns=["label"]
    )

    y_text = final_df["label"].astype(str)

    # ========================================================
    # Label encoder
    # ========================================================

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_text
    )

    gesture_classes = list(
        label_encoder.classes_
    )

    print(
        "Number of classes:",
        len(gesture_classes)
    )

    print(
        "Classes:",
        gesture_classes
    )

    # ========================================================
    # Save feature columns
    # ========================================================

    feature_columns = list(
        X.columns
    )

    joblib.dump(
        feature_columns,
        FEATURES_FILE
    )

    # Save classes
    joblib.dump(
        gesture_classes,
        CLASSES_FILE
    )

    # ========================================================
    # Train / validation / test
    # ========================================================

    X_train_temp, X_temp, y_train_temp, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
    )

    print(
        "Training samples:",
        len(X_train_temp)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # ========================================================
    # XGBoost
    #
    # Same hyperparameters from your successful training.
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(gesture_classes),

        colsample_bytree=0.7,
        gamma=1,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        n_estimators=200,
        subsample=0.8,

        reg_alpha=0.2,
        reg_lambda=0.8,

        eval_metric="mlogloss",

        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train_temp,
        y_train_temp
    )

    # ========================================================
    # Validation
    # ========================================================

    val_pred = xgb_model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    # ========================================================
    # Test
    # ========================================================

    test_pred = xgb_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=gesture_classes
        )
    )

    # ========================================================
    # Save model
    # ========================================================

    joblib.dump(
        xgb_model,
        MODEL_FILE
    )

    print(
        f"\nModel saved:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"Classes saved:"
        f"\n{CLASSES_FILE}"
    )

    print(
        f"Features saved:"
        f"\n{FEATURES_FILE}"
    )

    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

    model = xgb_model

    return xgb_model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model
    global feature_columns
    global gesture_classes

    # --------------------------------------------------------
    # Model does not exist
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):

        print(
            "\nWARNING:"
            "\nbest_xgboost.joblib not found."
            "\nTraining model from CSV..."
        )

        return train_model()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading trained XGBoost model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load feature columns
    # --------------------------------------------------------

    if os.path.exists(FEATURES_FILE):

        feature_columns = joblib.load(
            FEATURES_FILE
        )

    elif hasattr(model, "feature_names_in_"):

        feature_columns = list(
            model.feature_names_in_
        )

    else:

        feature_columns = None

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    if os.path.exists(CLASSES_FILE):

        gesture_classes = joblib.load(
            CLASSES_FILE
        )

    elif hasattr(model, "classes_"):

        # Numeric XGBoost classes
        gesture_classes = GESTURE_CLASSES

    else:

        gesture_classes = GESTURE_CLASSES

    print(
        "Model loaded successfully."
    )

    print(
        "Number of features:",
        len(feature_columns)
        if feature_columns is not None
        else "Unknown"
    )

    print(
        "Number of classes:",
        len(gesture_classes)
    )

    return model


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ============================================================
# EXTRACT HAND LANDMARKS
# ============================================================

def extract_hand_landmarks(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        image_rgb
    )

    if not results.multi_hand_landmarks:

        return None

    hand_landmarks = (
        results.multi_hand_landmarks[0]
    )

    landmarks = {}

    # MediaPipe gives 21 landmarks
    # 0 -> 20
    # We store as x1 -> x21
    for i, landmark in enumerate(
        hand_landmarks.landmark
    ):

        landmarks[
            f"x{i + 1}"
        ] = landmark.x

        landmarks[
            f"y{i + 1}"
        ] = landmark.y

        # z is not required for this model

    return pd.DataFrame(
        [landmarks]
    )


# ============================================================
# PREPARE IMAGE FEATURES
# ============================================================

def prepare_image_features(image):

    raw_df = extract_hand_landmarks(
        image
    )

    if raw_df is None:

        return None

    # Normalize
    normalized = normalize_coordinates(
        raw_df
    )

    # Add engineered features
    enhanced = calculate_finger_tip_distances(
        normalized
    )

    # --------------------------------------------------------
    # Match training feature order
    # --------------------------------------------------------

    if feature_columns is not None:

        for column in feature_columns:

            if column not in enhanced.columns:

                enhanced[column] = 0.0

        enhanced = enhanced[
            feature_columns
        ]

    return enhanced


# ============================================================
# PREDICT GESTURE
# ============================================================

def predict_gesture(image):

    if model is None:

        load_model()

    features = prepare_image_features(
        image
    )

    if features is None:

        return {
            "gesture": "No Hand Detected",
            "confidence": 0.0
        }

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )[0]

    best_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[best_index]
    )

    # --------------------------------------------------------
    # Get class name
    # --------------------------------------------------------

    if (
        hasattr(model, "classes_")
        and len(model.classes_) > best_index
    ):

        encoded_class = model.classes_[
            best_index
        ]

        try:

            encoded_class = int(
                encoded_class
            )

            if (
                0 <= encoded_class
                < len(gesture_classes)
            ):

                gesture = gesture_classes[
                    encoded_class
                ]

            else:

                gesture = gesture_classes[
                    best_index
                ]

        except Exception:

            gesture = gesture_classes[
                best_index
            ]

    else:

        gesture = gesture_classes[
            best_index
        ]

    return {
        "gesture": gesture,
        "confidence": round(
            confidence,
            4
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Hand Gesture Detection API is running",
        "model": "XGBoost",
        "classes": gesture_classes
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        file = request.files[
            "image"
        ]

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        result = predict_gesture(
            image
        )

        return jsonify({
            "success": True,
            "gesture": result["gesture"],
            "confidence": result["confidence"]
        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STARTUP
# ============================================================

print("=" * 60)
print("HAND GESTURE DETECTION SERVER")
print("=" * 60)

try:

    load_model()

except Exception as e:

    print(
        "\nMODEL INITIALIZATION ERROR:"
    )

    print(
        str(e)
    )

    # Do not crash immediately.
    # Flask can still start and show the error.
    model = None


# ============================================================
# LOCAL / RENDER SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"\nServer starting on port {port}..."
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
