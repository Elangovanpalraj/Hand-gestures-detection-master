import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Data Preprocessing & Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# Evaluation Metrics
from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      f1_score,
      classification_report,
      confusion_matrix
  )
# read model file
import joblib
# Real-time Video & Hand Tracking
import cv2
# Smoothing & Utilities
from collections import deque



# =============================================================================================
df  = pd.read_csv(r"F:\Hand-gestures-detection-master\hand_landmarks_data.csv")
#print(df.head())
# =============================================================================================

# Select a single sample (randomly or first row)
sample = df.iloc[0, :-1].values  # Exclude label column

# Extract x and y coordinates
x_coords = sample[0::3]  # extract all Xs
y_coords = sample[1::3]  # extract all Ys

# Plot hand landmarks
plt.figure(figsize=(5, 5))
plt.scatter(x_coords, y_coords, color="blue")
plt.plot(x_coords, y_coords, linestyle='-', marker='o')  # Connect keypoints

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Raw Hand Landmarks Before Normalization")
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
# plt.show()
# ===================================================================================================
def normalize_landmarks(sample):
    # Extract x and y coordinates
    x_coords = np.array(sample[0::3])
    y_coords = np.array(sample[1::3])

    # Use wrist (landmark 0) as reference point
    wrist_x, wrist_y = x_coords[0], y_coords[0]

    # Recenter the hand
    x_coords -= wrist_x
    y_coords -= wrist_y

    # Scale using the mid-finger tip (landmark 12 or 8)
    scale_factor = np.linalg.norm([x_coords[12], y_coords[12]])  # Distance to mid-finger tip

    # Avoid division by zero
    if scale_factor > 0:
        x_coords /= scale_factor
        y_coords /= scale_factor

    return x_coords, y_coords

# Normalize and visualize
x_norm, y_norm = normalize_landmarks(sample)

plt.figure(figsize=(5, 5))
plt.scatter(x_norm, y_norm, color="red")
plt.plot(x_norm, y_norm, linestyle='-', marker='o')

plt.xlabel("Normalized X Coordinate")
plt.ylabel("Normalized Y Coordinate")
plt.title("Normalized Hand Landmarks")
plt.gca().invert_yaxis()
# plt.show()
# ===============================================================================

# Drop all columns with 'z' in their names
df_2D = df.drop(columns=[col for col in df.columns if 'z' in col])
# =============================================================================
#convert label column into string type
df_2D["label"] = df_2D["label"].astype("string")
df_2D["label"].nunique()
# ====================================================================================
# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df_2D.iloc[:, -1])

# Change the column type before assigning
df_2D.iloc[:, -1] = pd.Series(encoded_labels, index=df_2D.index, dtype="int32")

# Save the mapping
label_mapping = dict(zip(label_encoder.classes_, encoded_labels))
# ==========================================================================================
# Select x and y coordinate columns separately
x_coords = [col for col in df_2D.columns if "x" in col]
y_coords = [col for col in df_2D.columns if "y" in col]

# Extract x and y values
x_vals = df_2D[x_coords]
y_vals = df_2D[y_coords]

# Get wrist coordinates (landmark 0, which is the first column in x_vals/y_vals)
wrist_x = x_vals.iloc[:, 0]
wrist_y = y_vals.iloc[:, 0]

# Recenter hand coordinates so that the wrist is at the origin
x_vals = x_vals.subtract(wrist_x, axis=0)
y_vals = y_vals.subtract(wrist_y, axis=0)

# Compute the distance from wrist to middle finger tip (landmark 13 used here because indexing starts from 1)
# This distance is used to normalize hand size (scale-invariant features)
dist_wrist_to_fingertip = np.sqrt((x_vals['x13'] - wrist_x)**2 + (y_vals['y13'] - wrist_y)**2)

# Normalize each coordinate by dividing by the wrist-to-fingertip distance (per sample)
# This ensures scale normalization across different hand sizes or camera distances
for i in range(len(x_vals)):
    if dist_wrist_to_fingertip[i] > 0:
        x_vals.iloc[i] /= dist_wrist_to_fingertip[i]
        y_vals.iloc[i] /= dist_wrist_to_fingertip[i]

# Combine x and y values column-wise, interleaved as x1, y1, x2, y2, ...
normalized_data = []
for x_col, y_col in zip(x_vals.columns, y_vals.columns):
    normalized_data.append(x_vals[x_col])
    normalized_data.append(y_vals[y_col])

# Create the final normalized DataFrame from interleaved x and y columns
normalized_df = pd.concat(normalized_data, axis=1)

# Add the label column back to the normalized DataFrame
normalized_df = pd.concat([normalized_df, df_2D["label"]], axis=1)
# ======================================================================================
print(normalized_df.head())
# ========================================================================================
import seaborn as sns
import matplotlib.pyplot as plt

class_counts = df_2D.iloc[:, -1].value_counts()

sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", legend=False)

plt.xlabel("Class Label")
plt.ylabel("Count")
plt.title("Class Distribution")
# plt.show()
# ===================================================================================================
# Select one sample to visualize
sample = normalized_df.iloc[0, :-1].values.reshape(-1, 2)

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(sample[:, 0], sample[:, 1], c="blue", marker="o")  # X vs Y

for i, txt in enumerate(range(len(sample))):
    plt.annotate(txt, (sample[i, 0], sample[i, 1]), fontsize=8, color="red")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("2D Hand Landmark Visualization")
# plt.show()

#  ========================================================================================

plt.figure(figsize=(12, 6))
sns.boxplot(data=normalized_df.iloc[:, 1:-1])  # Skip first and last columns
plt.xticks(rotation=90)
plt.title("Box Plot of Hand Landmarks (X and Y Coordinates)")
# plt.show()
# ====================================================================================================
sns.pairplot(normalized_df.iloc[:, 1:10])  # Select a few columns to avoid clutter
# plt.show()
# =======================================================================================
# Compute correlation matrix (excluding first column if needed)
corr_matrix = normalized_df.iloc[:, 1:].corr()

# Display correlation values
print(corr_matrix)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Hand Landmark Correlations")
# plt.show()
# =================================================================================================

pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_df.iloc[:, :-1])  # Exclude label column

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=normalized_df['label'], palette="tab10", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Hand Landmarks")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
# plt.show()
# ==================================================================================================
# Separate features (X) and label (y)
X = normalized_df.drop(columns=["label"])  # Features
y = normalized_df["label"]  # Target

# ===================================================================================
# Split the dataset into training,val,test
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# ===========================================================================================
# Initialize individual models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Initialize the ensemble model (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('rf', models['Random Forest']),
    ('svm', models['SVM']),
    ('xgb', models['XGBoost']),
    ('lr', models['Logistic Regression']),
    ('dt', models['Decision Tree'])
], voting='hard')  # 'hard' voting for classification (majority rule)

# Add ensemble to models dictionary for easier handling
models['Ensemble'] = ensemble_model

# Set up K-Fold Cross Validation
kf = KFold(n_splits=7, shuffle=True, random_state=42)

# Dictionary to store results
results = {}

# Perform Cross-Validation for each model
for name, model in models.items():
    print(f"Training and evaluating {name}...")

    # Cross-validation score (accuracy)
    scores = cross_val_score(model, X_train_temp, y_train_temp, cv=kf, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    std_dev = np.std(scores)

    # Store the results
    results[name] = {
        'mean_accuracy': mean_accuracy,
        'std_dev': std_dev
    }

    # Train the model on the full training data
    model.fit(X_train_temp, y_train_temp)

    # Predictions on the training set
    y_pred_train = model.predict(X_train_temp)

    # Compute evaluation metrics for training data
    accuracy = accuracy_score(y_train_temp, y_pred_train)
    precision = precision_score(y_train_temp, y_pred_train, average='macro')
    recall = recall_score(y_train_temp, y_pred_train, average='macro')
    f1 = f1_score(y_train_temp, y_pred_train, average='macro')

    # Add metrics to results
    results[name].update({
        'train_accuracy': accuracy,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1
    })

    print(f"{name} (Train) - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Evaluate models on the validation set
for name, model in models.items():
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, average='macro')
    val_recall = recall_score(y_val, y_pred_val, average='macro')
    val_f1 = f1_score(y_val, y_pred_val, average='macro')

    results[name].update({
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    })

    print(f"{name} Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

# Final evaluation on the test set
for name, model in models.items():
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    test_recall = recall_score(y_test, y_pred_test, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    results[name].update({
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    })

    print(f"{name} Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# Conclusion: Summarizing the best model
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model_results = results[best_model_name]

print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall: {metrics['test_recall']:.4f}")
    print(f"  F1-score: {metrics['test_f1']:.4f}")

print(f"\nBest Performing Model: {best_model_name}")
print(f"Test Accuracy: {best_model_results['test_accuracy']:.4f}")
print(f"Test Precision: {best_model_results['test_precision']:.4f}")
print(f"Test Recall: {best_model_results['test_recall']:.4f}")
print(f"Test F1-score: {best_model_results['test_f1']:.4f}")
# ============================================================================================

#hyperparameter tuning before adding new features
# Split the dataset into training, validation, and test sets
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the XGBoost model with regularization
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Define hyperparameter grid for tuning (with additional values to prevent overfitting)
param_grid = {
    'learning_rate': [0.1, 0.05],  # Reduced learning rate to avoid overfitting
    'n_estimators': [50, 100],  # Keep number of estimators moderate
    'subsample': [0.8, 0.9],  # Introduced higher subsample to prevent overfitting
    'colsample_bytree': [0.8],  # Keep one value to simplify
    'max_depth': [5, 6],  # Added a larger value to prevent underfitting
    'min_child_weight': [1, 3],  # Reduced values to avoid overfitting
    'gamma': [0, 0.1],  # Added small gamma to make the model more conservative
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

# Fit the GridSearchCV on the training data
grid_search.fit(X_train_temp, y_train_temp)

# Print all results per run
print("\n=== Grid Search Results ===")
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(f"Params: {params}, Mean Accuracy: {mean_score:.4f}")

# Get the best hyperparameters
print(f"\nBest Parameters: {grid_search.best_params_}")

# Train the best model from grid search
best_model = grid_search.best_estimator_

# Function to calculate precision and F1 for multiclass (weighted average)
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # Weighted average for multiclass
    f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted average for multiclass
    return accuracy, precision, f1

# Evaluate on training set
train_pred = best_model.predict(X_train_temp)
train_accuracy, train_precision, train_f1 = calculate_metrics(y_train_temp, train_pred)

print("\nTraining Set Evaluation:")
print(f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, F1-Score: {train_f1:.4f}")

# Evaluate on validation set
val_pred = best_model.predict(X_val)
val_accuracy, val_precision, val_f1 = calculate_metrics(y_val, val_pred)

print("\nValidation Set Evaluation:")
print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, F1-Score: {val_f1:.4f}")

# Final evaluation on test set
test_pred = best_model.predict(X_test)
test_accuracy, test_precision, test_f1 = calculate_metrics(y_test, test_pred)

print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, F1-Score: {test_f1:.4f}")

# K-Fold Cross-Validation (with the best model)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_temp, y_train_temp, cv=kf, scoring='accuracy')

print(f"\nXGBoost K-Fold Cross-Validation Mean Accuracy: {np.mean(cv_scores):.4f}, Std Dev = {np.std(cv_scores):.4f}")
# =======================================================================================================================
def calculate_finger_tip_distances(df):
    """
    Calculate distances between finger tips and create new features
    Args:
        df: DataFrame with columns x1,y1,x2,y2,...,x20,y20 (indices 1-20)
    Returns:
        DataFrame with new distance features added
    """
    df = df.copy()

    # Define indices according to YOUR dataset (1-20)
    TIP_INDICES = {
        'thumb': 5,   # Changed from original 4 to match your data
        'index': 9,   # Changed from 8
        'middle': 13, # Changed from 12
        'ring': 17,   # Changed from 16
        'pinky': 21   # Changed from 20
    }

    FINGER_BASES = {
        'thumb': 2,   # Base of thumb
        'index': 6,   # Knuckle of index
        'middle': 10,  # Knuckle of middle
        'ring': 14,   # Knuckle of ring
        'pinky': 18   # Knuckle of pinky
    }

    # 1. Calculate all pairwise tip distances
    tips = list(TIP_INDICES.values())
    for i, tip1 in enumerate(tips[:-1]):
        for tip2 in tips[i+1:]:
            # Calculate Euclidean distance
            dist = np.sqrt(
                (df[f'x{tip1}'] - df[f'x{tip2}'])**2 +
                (df[f'y{tip1}'] - df[f'y{tip2}'])**2
            )

            # Name the feature
            name1 = [k for k,v in TIP_INDICES.items() if v == tip1][0]
            name2 = [k for k,v in TIP_INDICES.items() if v == tip2][0]
            df[f'dist_{name1}_{name2}'] = dist

    # 2. Thumb-to-other distances
    thumb_x, thumb_y = df['x5'], df['y5']  # Using your thumb tip index
    for finger in ['index', 'middle', 'ring', 'pinky']:
        tip = TIP_INDICES[finger]
        df[f'dist_thumb_{finger}'] = np.sqrt(
            (thumb_x - df[f'x{tip}'])**2 +
            (thumb_y - df[f'y{tip}'])**2
        )

    # 3. Finger lengths (base to tip)
    for finger, base in FINGER_BASES.items():
        tip = TIP_INDICES[finger]
        df[f'len_{finger}'] = np.sqrt(
            (df[f'x{base}'] - df[f'x{tip}'])**2 +
            (df[f'y{base}'] - df[f'y{tip}'])**2
        )

    return df

# ===========================================================================================================
df_features = normalized_df.drop(columns=['label'])
# ===========================================================================================================
# Add features
enhanced_df = calculate_finger_tip_distances(df_features)
final_df = pd.concat([enhanced_df,normalized_df["label"] ], axis=1)
print(final_df.head())
#  ==============================================================================================================
# Separate features (X) and label (y)
X = final_df.drop(columns=["label"])  # Features
y = final_df["label"]  # Target
# Split the dataset into training,val,test
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# =====================================================================================================
# Initialize individual models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Initialize the ensemble model (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('rf', models['Random Forest']),
    ('svm', models['SVM']),
    ('xgb', models['XGBoost']),
    ('lr', models['Logistic Regression']),
    ('dt', models['Decision Tree'])
], voting='hard')  # 'hard' voting for classification (majority rule)

# Add ensemble to models dictionary for easier handling
models['Ensemble'] = ensemble_model

# Set up K-Fold Cross Validation
kf = KFold(n_splits=7, shuffle=True, random_state=42)

# Dictionary to store results
results = {}

# Perform Cross-Validation for each model
for name, model in models.items():
    print(f"Training and evaluating {name}...")

    # Cross-validation score (accuracy)
    scores = cross_val_score(model, X_train_temp, y_train_temp, cv=kf, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    std_dev = np.std(scores)

    # Store the results
    results[name] = {
        'mean_accuracy': mean_accuracy,
        'std_dev': std_dev
    }

    # Train the model on the full training data
    model.fit(X_train_temp, y_train_temp)

    # Predictions on the training set
    y_pred_train = model.predict(X_train_temp)

    # Compute evaluation metrics for training data
    accuracy = accuracy_score(y_train_temp, y_pred_train)
    precision = precision_score(y_train_temp, y_pred_train, average='macro')
    recall = recall_score(y_train_temp, y_pred_train, average='macro')
    f1 = f1_score(y_train_temp, y_pred_train, average='macro')

    # Add metrics to results
    results[name].update({
        'train_accuracy': accuracy,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1
    })

    print(f"{name} (Train) - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Evaluate models on the validation set
for name, model in models.items():
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, average='macro')
    val_recall = recall_score(y_val, y_pred_val, average='macro')
    val_f1 = f1_score(y_val, y_pred_val, average='macro')

    results[name].update({
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    })

    print(f"{name} Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

# Final evaluation on the test set
for name, model in models.items():
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    test_recall = recall_score(y_test, y_pred_test, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    results[name].update({
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    })

    print(f"{name} Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# Conclusion: Summarizing the best model
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model_results = results[best_model_name]

print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall: {metrics['test_recall']:.4f}")
    print(f"  F1-score: {metrics['test_f1']:.4f}")

print(f"\nBest Performing Model: {best_model_name}")
print(f"Test Accuracy: {best_model_results['test_accuracy']:.4f}")
print(f"Test Precision: {best_model_results['test_precision']:.4f}")
print(f"Test Recall: {best_model_results['test_recall']:.4f}")
print(f"Test F1-score: {best_model_results['test_f1']:.4f}")
# ====================================================================================
# Split the dataset into training, validation, and test sets
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the XGBoost model with regularization
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Define hyperparameter grid for tuning (with additional values to prevent overfitting)
param_grid = {
    'colsample_bytree': [0.7],
    'gamma': [1],
    'learning_rate': [0.05],
    'max_depth': [4],
    'min_child_weight': [5],
    'n_estimators': [200],
    'subsample': [0.8],
    'reg_alpha': [0.2],
    'reg_lambda': [0.8]
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

# Fit the GridSearchCV on the training data
grid_search.fit(X_train_temp, y_train_temp)

# Print all results per run
print("\n=== Grid Search Results ===")
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(f"Params: {params}, Mean Accuracy: {mean_score:.4f}")

# Get the best hyperparameters
print(f"\nBest Parameters: {grid_search.best_params_}")

# Train the best model from grid search
best_model = grid_search.best_estimator_

# Function to calculate precision and F1 for multiclass (weighted average)
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # Weighted average for multiclass
    f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted average for multiclass
    return accuracy, precision, f1

# Evaluate on training set
train_pred = best_model.predict(X_train_temp)
train_accuracy, train_precision, train_f1 = calculate_metrics(y_train_temp, train_pred)

print("\nTraining Set Evaluation:")
print(f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, F1-Score: {train_f1:.4f}")

# Evaluate on validation set
val_pred = best_model.predict(X_val)
val_accuracy, val_precision, val_f1 = calculate_metrics(y_val, val_pred)

print("\nValidation Set Evaluation:")
print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, F1-Score: {val_f1:.4f}")

# Final evaluation on test set
test_pred = best_model.predict(X_test)
test_accuracy, test_precision, test_f1 = calculate_metrics(y_test, test_pred)

print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, F1-Score: {test_f1:.4f}")

# K-Fold Cross-Validation (with the best model)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_temp, y_train_temp, cv=kf, scoring='accuracy')

print(f"\nXGBoost K-Fold Cross-Validation Mean Accuracy: {np.mean(cv_scores):.4f}, Std Dev = {np.std(cv_scores):.4f}")
# ======================================================================================================================
# Save the best model to a file
model_filename = 'best_xgboost.joblib'
joblib.dump(best_model, model_filename)
print(f"\nModel saved to {model_filename}")
# =========================================================================================================
loaded_model = joblib.load('best_xgboost.joblib')
# Make predictions using the loaded model
y_pred = loaded_model.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred)

print(f"Final Test Accuracy of the Loaded Model: {final_accuracy:.4f}")
# ============================================================================================================
# Load trained XGBoost model
model = joblib.load("best_xgboost.joblib")

# Gesture classes
gesture_classes = [
    'call', 'dislike', 'fist', 'four', 'like',
    'mute', 'ok', 'one', 'palm', 'peace',
    'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def preprocess_data(df):
    df_2D = df.drop(columns=[col for col in df.columns if 'z' in col])
    return df_2D

def normalize_coordinates(df):
    x_coords = [col for col in df.columns if "x" in col]
    y_coords = [col for col in df.columns if "y" in col]
    x_vals, y_vals = df[x_coords], df[y_coords]
    wrist_x, wrist_y = x_vals.iloc[:, 0], y_vals.iloc[:, 0]
    x_vals, y_vals = x_vals.subtract(wrist_x, axis=0), y_vals.subtract(wrist_y, axis=0)
    dist_wrist_to_fingertip = np.sqrt((x_vals['x13'] - wrist_x)**2 + (y_vals['y13'] - wrist_y)**2)
    for i in range(len(x_vals)):
        if dist_wrist_to_fingertip[i] > 0:
            x_vals.iloc[i] /= dist_wrist_to_fingertip[i]
            y_vals.iloc[i] /= dist_wrist_to_fingertip[i]

    normalized_data = []
    for x_col, y_col in zip(x_vals.columns, y_vals.columns):
        normalized_data.append(x_vals[x_col])
        normalized_data.append(y_vals[y_col])

    normalized_df = pd.concat(normalized_data, axis=1)
    if "label" in df.columns:
        normalized_df = pd.concat([normalized_df, df["label"]], axis=1)

    return normalized_df

def calculate_finger_tip_distances(df):
    df = df.copy()
    TIP_INDICES = {'thumb': 5, 'index': 9, 'middle': 13, 'ring': 17, 'pinky': 21}
    FINGER_BASES = {'thumb': 2, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}

    tips = list(TIP_INDICES.values())
    for i, tip1 in enumerate(tips[:-1]):
        for tip2 in tips[i+1:]:
            dist = np.sqrt((df[f'x{tip1}'] - df[f'x{tip2}'])**2 + (df[f'y{tip1}'] - df[f'y{tip2}'])**2)
            name1 = [k for k,v in TIP_INDICES.items() if v == tip1][0]
            name2 = [k for k,v in TIP_INDICES.items() if v == tip2][0]
            df[f'dist_{name1}_{name2}'] = dist

    thumb_x, thumb_y = df['x5'], df['y5']
    for finger in ['index', 'middle', 'ring', 'pinky']:
        tip = TIP_INDICES[finger]
        df[f'dist_thumb_{finger}'] = np.sqrt((thumb_x - df[f'x{tip}'])**2 + (thumb_y - df[f'y{tip}'])**2)

    for finger, base in FINGER_BASES.items():
        tip = TIP_INDICES[finger]
        df[f'len_{finger}'] = np.sqrt((df[f'x{base}'] - df[f'x{tip}'])**2 + (df[f'y{base}'] - df[f'y{tip}'])**2)

    return df

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = {}
    for i, landmark in enumerate(hand_landmarks.landmark):
        landmarks[f'x{i+1}'] = landmark.x
        landmarks[f'y{i+1}'] = landmark.y
    return pd.DataFrame([landmarks])

def predict_gesture(image):
    df = extract_hand_landmarks(image)
    if df is None:
        return "No Hand Detected", 0.0
    df = preprocess_data(df)
    df = normalize_coordinates(df)
    df = calculate_finger_tip_distances(df)
    probabilities = model.predict_proba(df)[0]
    best_idx = np.argmax(probabilities)
    gesture = gesture_classes[best_idx]
    confidence = float(probabilities[best_idx])
    return gesture, confidence

# Initialize video capture and writer
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('gesture_recognition_output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Extract hand landmarks and predict gesture
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gesture, confidence = predict_gesture(frame)
    if gesture == "No Hand Detected":
        display_text = gesture
    else:
        display_text = f"{gesture}: {confidence:.2f}"

    # Display the text on the frame
    cv2.putText(frame, f'Gesture: {display_text}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw hand landmarks with thinner lines
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))  # Thinner lines

    # Write the frame to the video file
    out.write(frame)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()












