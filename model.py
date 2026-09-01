# ============================================================
# Hand Gesture Recognition - Real Time Model
# ============================================================

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_FILE = "best_xgboost.joblib"
CLASSES_FILE = "gesture_classes.joblib"
FEATURES_FILE = "feature_columns.joblib"

OUTPUT_VIDEO = "gesture_recognition_output.avi"


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    CLASSES_FILE,
    FEATURES_FILE
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"\nERROR: Required file not found: {file}\n"
            f"Please make sure {file} is present in the project folder."
        )


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("Loading trained XGBoost model...")

model = joblib.load(MODEL_FILE)
gesture_classes = joblib.load(CLASSES_FILE)
feature_columns = joblib.load(FEATURES_FILE)

print("Model loaded successfully.")
print("Number of classes:", len(gesture_classes))
print("Number of features:", len(feature_columns))

print("\nGesture Classes:")
for i, gesture in enumerate(gesture_classes):
    print(f"{i}: {gesture}")


# ============================================================
# 4. MEDIAPIPE INITIALIZATION
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
# 5. PREPROCESS LANDMARK DATA
# ============================================================

def preprocess_data(df):
    """
    Remove Z coordinates.

    Input:
        x1, y1, z1, x2, y2, z2, ...

    Output:
        x1, y1, x2, y2, ...
    """

    df = df.copy()

    z_columns = [
        col for col in df.columns
        if col.lower().startswith("z")
        or "_z" in col.lower()
    ]

    if z_columns:
        df = df.drop(columns=z_columns)

    return df


# ============================================================
# 6. NORMALIZE HAND COORDINATES
# ============================================================

def normalize_coordinates(df):
    """
    Normalize hand landmarks.

    Steps:
    1. Wrist becomes origin.
    2. Scale using wrist -> middle finger tip.
    3. Return interleaved x,y coordinates.

    Output:
        x1,y1,x2,y2,...,x21,y21
    """

    df = df.copy()

    # --------------------------------------------------------
    # Get X and Y columns
    # --------------------------------------------------------

    x_coords = sorted(
        [col for col in df.columns if col.lower().startswith("x")],
        key=lambda x: int(x[1:])
    )

    y_coords = sorted(
        [col for col in df.columns if col.lower().startswith("y")],
        key=lambda x: int(x[1:])
    )

    if len(x_coords) != 21 or len(y_coords) != 21:
        raise ValueError(
            f"Expected 21 X and 21 Y landmarks, "
            f"but got {len(x_coords)} X and {len(y_coords)} Y."
        )

    # --------------------------------------------------------
    # Copy numeric values
    # --------------------------------------------------------

    x_vals = df[x_coords].astype(float).copy()
    y_vals = df[y_coords].astype(float).copy()

    # --------------------------------------------------------
    # Wrist = landmark 1 in dataset naming
    # --------------------------------------------------------

    wrist_x = x_vals["x1"].copy()
    wrist_y = y_vals["y1"].copy()

    # --------------------------------------------------------
    # Move wrist to (0,0)
    # --------------------------------------------------------

    x_vals = x_vals.subtract(wrist_x, axis=0)
    y_vals = y_vals.subtract(wrist_y, axis=0)

    # --------------------------------------------------------
    # Scale using wrist -> middle fingertip
    #
    # Dataset:
    # x13,y13 = middle finger tip
    # --------------------------------------------------------

    scale_factor = np.sqrt(
        x_vals["x13"] ** 2 +
        y_vals["y13"] ** 2
    )

    # Avoid division by zero
    scale_factor = scale_factor.replace(0, 1.0)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    x_vals = x_vals.div(scale_factor, axis=0)
    y_vals = y_vals.div(scale_factor, axis=0)

    # --------------------------------------------------------
    # Interleave:
    #
    # x1,y1,x2,y2,...,x21,y21
    # --------------------------------------------------------

    normalized_data = {}

    for i in range(21):

        x_col = f"x{i + 1}"
        y_col = f"y{i + 1}"

        normalized_data[x_col] = x_vals[x_col]
        normalized_data[y_col] = y_vals[y_col]

    normalized_df = pd.DataFrame(
        normalized_data,
        index=df.index
    )

    return normalized_df


# ============================================================
# 7. ADD FINGER DISTANCE FEATURES
# ============================================================

def calculate_finger_tip_distances(df):
    """
    Add 15 additional geometric features:

    10 pairwise finger-tip distances
    5 finger lengths

    Total:
        42 landmark features
        + 10 tip distances
        + 5 finger lengths
        = 57 features
    """

    df = df.copy()

    # --------------------------------------------------------
    # Finger tips
    # --------------------------------------------------------

    TIP_INDICES = {
        "thumb": 5,
        "index": 9,
        "middle": 13,
        "ring": 17,
        "pinky": 21
    }

    # --------------------------------------------------------
    # Finger bases
    # --------------------------------------------------------

    FINGER_BASES = {
        "thumb": 2,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18
    }

    # ========================================================
    # 7.1 Pairwise tip distances
    #
    # 5 fingers -> 10 unique combinations
    # ========================================================

    finger_names = list(TIP_INDICES.keys())

    for i in range(len(finger_names)):

        finger1 = finger_names[i]
        tip1 = TIP_INDICES[finger1]

        for j in range(i + 1, len(finger_names)):

            finger2 = finger_names[j]
            tip2 = TIP_INDICES[finger2]

            distance = np.sqrt(
                (df[f"x{tip1}"] - df[f"x{tip2}"]) ** 2 +
                (df[f"y{tip1}"] - df[f"y{tip2}"]) ** 2
            )

            feature_name = f"dist_{finger1}_{finger2}"

            df[feature_name] = distance

    # ========================================================
    # 7.2 Finger lengths
    #
    # 5 additional features
    # ========================================================

    for finger, base in FINGER_BASES.items():

        tip = TIP_INDICES[finger]

        length = np.sqrt(
            (df[f"x{base}"] - df[f"x{tip}"]) ** 2 +
            (df[f"y{base}"] - df[f"y{tip}"]) ** 2
        )

        feature_name = f"len_{finger}"

        df[feature_name] = length

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT add thumb-to-other distances separately.
    #
    # They are already included in pairwise tip distances.
    #
    # This keeps:
    #
    # 42 + 10 + 5 = 57 features
    # --------------------------------------------------------

    return df


# ============================================================
# 8. EXTRACT HAND LANDMARKS
# ============================================================

def extract_hand_landmarks(image):
    """
    Detect hand and extract 21 MediaPipe landmarks.

    Returns:
        DataFrame with x1,y1,z1,...,x21,y21,z21
    """

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = {}

    for i, landmark in enumerate(hand_landmarks.landmark):

        landmark_number = i + 1

        landmarks[f"x{landmark_number}"] = landmark.x
        landmarks[f"y{landmark_number}"] = landmark.y
        landmarks[f"z{landmark_number}"] = landmark.z

    df = pd.DataFrame([landmarks])

    return df, hand_landmarks


# ============================================================
# 9. CREATE FINAL FEATURES FOR MODEL
# ============================================================

def create_features(image):
    """
    Complete preprocessing pipeline:

    Image
      ↓
    MediaPipe landmarks
      ↓
    Remove Z
      ↓
    Normalize
      ↓
    Add distance features
      ↓
    Match training feature columns
      ↓
    57 features
    """

    raw_df, hand_landmarks = extract_hand_landmarks(image)

    if raw_df is None:
        return None, None

    # --------------------------------------------------------
    # Remove Z
    # --------------------------------------------------------

    df = preprocess_data(raw_df)

    # --------------------------------------------------------
    # Normalize X/Y
    # --------------------------------------------------------

    df = normalize_coordinates(df)

    # --------------------------------------------------------
    # Add geometric features
    # --------------------------------------------------------

    df = calculate_finger_tip_distances(df)

    # --------------------------------------------------------
    # Make sure all expected training columns exist
    # --------------------------------------------------------

    missing_features = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nMissing features required by trained model:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Remove unexpected columns and preserve exact order
    # --------------------------------------------------------

    df = df[feature_columns]

    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if df.shape[1] != len(feature_columns):

        raise ValueError(
            f"\nFeature count mismatch!\n"
            f"Model expects: {len(feature_columns)}\n"
            f"Generated: {df.shape[1]}"
        )

    return df, hand_landmarks


# ============================================================
# 10. PREDICT GESTURE
# ============================================================

def predict_gesture(image):

    try:

        feature_df, hand_landmarks = create_features(image)

        # ----------------------------------------------------
        # No hand
        # ----------------------------------------------------

        if feature_df is None:

            return (
                "No Hand Detected",
                0.0,
                None
            )

        # ----------------------------------------------------
        # Predict probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(feature_df)[0]

        # ----------------------------------------------------
        # Best class
        # ----------------------------------------------------

        best_idx = int(np.argmax(probabilities))

        confidence = float(
            probabilities[best_idx]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # gesture_classes was saved during training.
        # ----------------------------------------------------

        gesture = str(
            gesture_classes[best_idx]
        )

        return (
            gesture,
            confidence,
            hand_landmarks
        )

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return (
            "Prediction Error",
            0.0,
            None
        )


# ============================================================
# 11. CAMERA SETUP
# ============================================================

print("\nStarting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam. "
        "Check camera permission."
    )


# ============================================================
# 12. CAMERA RESOLUTION
# ============================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


# ============================================================
# 13. VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    20.0,
    (FRAME_WIDTH, FRAME_HEIGHT)
)


# ============================================================
# 14. REAL-TIME LOOP
# ============================================================

print("\n========================================")
print(" HAND GESTURE RECOGNITION STARTED")
print("========================================")
print("Press Q to quit.")
print("")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        print("Failed to read frame.")
        break

    # --------------------------------------------------------
    # Mirror image
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    gesture, confidence, hand_landmarks = predict_gesture(
        frame
    )

    # ========================================================
    # DISPLAY TEXT
    # ========================================================

    if gesture == "No Hand Detected":

        display_text = "No Hand Detected"

        color = (0, 0, 255)

    elif gesture == "Prediction Error":

        display_text = "Prediction Error"

        color = (0, 0, 255)

    else:

        display_text = (
            f"{gesture} "
            f"({confidence * 100:.1f}%)"
        )

        color = (0, 255, 0)

    # --------------------------------------------------------
    # Background rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (640, 80),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Gesture text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Gesture: {display_text}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # DRAW MEDIAPIPE LANDMARKS
    # ========================================================

    if hand_landmarks is not None:

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,

            mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            ),

            mp_drawing.DrawingSpec(
                color=(0, 0, 255),
                thickness=2,
                circle_radius=2
            )
        )

    # ========================================================
    # WRITE OUTPUT VIDEO
    # ========================================================

    out.write(frame)

    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    # --------------------------------------------------------
    # Press Q to exit
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 15. RELEASE EVERYTHING
# ============================================================

print("\nStopping camera...")

cap.release()
out.release()

cv2.destroyAllWindows()

hands.close()

print("Camera released.")
print(f"Output video saved as: {OUTPUT_VIDEO}")

print("\n========================================")
print(" PROGRAM FINISHED")
print("========================================")
