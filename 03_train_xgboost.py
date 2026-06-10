import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

# =====================================================
# OUTPUT FOLDERS
# =====================================================

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "outputs/cicids2017_clean.csv",
    low_memory=False
)

print("Dataset Shape:", df.shape)

# =====================================================
# REMOVE TINY CLASSES
# =====================================================

remove_classes = [
    "Heartbleed",
    "Infiltration",
    "Web Attack ï¿½ Sql Injection"
]

df = df[~df["Label"].isin(remove_classes)]

print("After Filtering:", df.shape)

# =====================================================
# LOAD TOP FEATURES
# =====================================================

importance = pd.read_csv(
    "outputs/tables/feature_importance.csv"
)

top_features = (
    importance
    .head(20)["Feature"]
    .tolist()
)

print("\nUsing Top 20 Features:\n")

for f in top_features:
    print(f)

# =====================================================
# LABEL ENCODING
# =====================================================

le = LabelEncoder()

df["Label"] = le.fit_transform(
    df["Label"]
)

# =====================================================
# FEATURES / TARGET
# =====================================================

X = df[top_features]

y = df["Label"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining XGBoost...")

start = time.time()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

end = time.time()

training_time = (end - start) / 60

print(
    f"\nTraining Time: {training_time:.2f} minutes"
)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nRESULTS")
print("="*50)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================================
# SAVE METRICS
# =====================================================

metrics = pd.DataFrame({
    "Accuracy":[accuracy],
    "Precision":[precision],
    "Recall":[recall],
    "F1":[f1],
    "Training_Time_Min":[training_time]
})

metrics.to_csv(
    "outputs/tables/metrics.csv",
    index=False
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(12,10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/confusion_matrix.png",
    dpi=300
)

plt.close()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "outputs/models/xgboost_ids.pkl"
)

print("\nSaved Files:")
print("outputs/models/xgboost_ids.pkl")
print("outputs/tables/metrics.csv")
print("outputs/figures/confusion_matrix.png")