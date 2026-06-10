import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("outputs/cicids2017_clean.csv")

# Remove tiny classes
remove_classes = [
    "Heartbleed",
    "Infiltration",
    "Web Attack ï¿½ Sql Injection"
]

df = df[~df["Label"].isin(remove_classes)]

print("Shape after filtering:", df.shape)

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Features and target
X = df.drop("Label", axis=1)
y = df["Label"]

# Train XGBoost
print("Training XGBoost for feature importance...")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

xgb.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# Save table
os.makedirs("outputs/tables", exist_ok=True)

importance.to_csv(
    "outputs/tables/feature_importance.csv",
    index=False
)

print("\nTop 20 Features:")
print(importance.head(20))

# Plot top 20
top20 = importance.head(20)

plt.figure(figsize=(10,8))

plt.barh(
    top20["Feature"][::-1],
    top20["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features - XGBoost")

os.makedirs("../outputs/figures", exist_ok=True)

plt.tight_layout()

plt.savefig(
    "outputs/figures/top20_features.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("outputs/tables/feature_importance.csv")
print("outputs/figures/top20_features.png")