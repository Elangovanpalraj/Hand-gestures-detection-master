import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "hand_landmarks_data.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# 2. REMOVE Z COORDINATES
# ============================================================

df_2D = df.drop(
    columns=[col for col in df.columns if "z" in col.lower()]
).copy()


# ============================================================
# 3. ENCODE LABELS
# ============================================================

df_2D["label"] = df_2D["label"].astype(str)

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(
    df_2D["label"]
)

# IMPORTANT:
# Do NOT assign integer values into a pandas StringDtype column.
df_2D["label"] = encoded_labels.astype(np.int32)

print("\nGesture classes:")
for number, gesture in enumerate(label_encoder.classes_):
    print(number, "=", gesture)


# ============================================================
# 4. NORMALIZE HAND COORDINATES
# ============================================================

x_coords = [
    col for col in df_2D.columns
    if col.startswith("x")
]

y_coords = [
    col for col in df_2D.columns
    if col.startswith("y")
]

x_vals = df_2D[x_coords].copy()
y_vals = df_2D[y_coords].copy()

# Wrist = landmark 1 in your dataset
wrist_x = x_vals.iloc[:, 0].copy()
wrist_y = y_vals.iloc[:, 0].copy()

# Move wrist to origin
x_vals = x_vals.subtract(wrist_x, axis=0)
y_vals = y_vals.subtract(wrist_y, axis=0)


# ============================================================
# 5. SCALE NORMALIZATION
# ============================================================

# Your original project uses landmark 13
dist_wrist_to_fingertip = np.sqrt(
    x_vals["x13"] ** 2 +
    y_vals["y13"] ** 2
)

# Avoid division by zero
dist_wrist_to_fingertip = dist_wrist_to_fingertip.replace(
    0,
    1e-8
)

x_vals = x_vals.div(
    dist_wrist_to_fingertip,
    axis=0
)

y_vals = y_vals.div(
    dist_wrist_to_fingertip,
    axis=0
)


# ============================================================
# 6. INTERLEAVE X AND Y
# ============================================================

normalized_data = {}

for x_col, y_col in zip(x_vals.columns, y_vals.columns):

    normalized_data[x_col] = x_vals[x_col]
    normalized_data[y_col] = y_vals[y_col]


normalized_df = pd.DataFrame(
    normalized_data,
    index=df_2D.index
)

normalized_df["label"] = df_2D["label"].astype(np.int32)


# ============================================================
# 7. FEATURE ENGINEERING
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

    # ----------------------------------------
    # Pairwise fingertip distances
    # ----------------------------------------

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

    # ----------------------------------------
    # Thumb to other fingertips
    # ----------------------------------------

    thumb_x = data["x5"]
    thumb_y = data["y5"]

    for finger in ["index", "middle", "ring", "pinky"]:

        tip = TIP_INDICES[finger]

        data[f"dist_thumb_{finger}"] = np.sqrt(
            (thumb_x - data[f"x{tip}"]) ** 2
            +
            (thumb_y - data[f"y{tip}"]) ** 2
        )

    # ----------------------------------------
    # Finger lengths
    # ----------------------------------------

    for finger, base in FINGER_BASES.items():

        tip = TIP_INDICES[finger]

        data[f"len_{finger}"] = np.sqrt(
            (data[f"x{base}"] - data[f"x{tip}"]) ** 2
            +
            (data[f"y{base}"] - data[f"y{tip}"]) ** 2
        )

    return data


# Remove label before adding features
features_only = normalized_df.drop(
    columns=["label"]
)

enhanced_df = calculate_finger_tip_distances(
    features_only
)

final_df = pd.concat(
    [
        enhanced_df,
        normalized_df["label"]
    ],
    axis=1
)


# ============================================================
# 8. X / Y
# ============================================================

X = final_df.drop(
    columns=["label"]
)

y = final_df["label"].astype(np.int32)


print("\nFinal feature shape:", X.shape)
print("Number of classes:", len(label_encoder.classes_))


# ============================================================
# 9. TRAIN / VALIDATION / TEST
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ============================================================
# 10. XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=1,
    reg_alpha=0.2,
    reg_lambda=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 12. VALIDATION
# ============================================================

val_prediction = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_prediction
)

print(
    f"\nValidation Accuracy: {val_accuracy:.4f}"
)


# ============================================================
# 13. TEST
# ============================================================

test_prediction = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_prediction
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_prediction,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "best_xgboost.joblib"
)

joblib.dump(
    label_encoder.classes_.tolist(),
    "gesture_classes.joblib"
)

# Save feature names/order too
joblib.dump(
    list(X.columns),
    "feature_columns.joblib"
)


print("\n======================================")
print("MODEL TRAINING COMPLETE")
print("======================================")
print("Created:")
print("  best_xgboost.joblib")
print("  gesture_classes.joblib")
print("  feature_columns.joblib")
