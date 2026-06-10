import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "outputs/cicids2017_clean.csv",
    low_memory=False
)

# Remove tiny classes

remove_classes = [
    "Heartbleed",
    "Infiltration",
    "Web Attack ï¿½ Sql Injection"
]

df = df[~df["Label"].isin(remove_classes)]

# Encode labels

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# =====================================================
# LOAD TOP FEATURES
# =====================================================

importance = pd.read_csv(
    "outputs/tables/feature_importance.csv"
)

top_features = (
    importance.head(20)["Feature"]
    .tolist()
)

print("\nTop Features Loaded:")
print(top_features)

# =====================================================
# SAMPLE DATA
# =====================================================

X = df[top_features]

X_sample = X.sample(
    n=1000,
    random_state=42
)

print("\nSample Shape:", X_sample.shape)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Model...")

model = joblib.load(
    "outputs/models/xgboost_ids.pkl"
)

# =====================================================
# SHAP EXPLAINER
# =====================================================

print("\nComputing SHAP Values...")

booster = model.get_booster()

explainer = shap.TreeExplainer(
    booster
)

shap_values = explainer(
    X_sample
)

print("SHAP Completed")

# =====================================================
# INSPECT SHAPE
# =====================================================

print("\nSHAP Shape:")

print(type(shap_values))

print(shap_values.values.shape)

# =====================================================
# HANDLE MULTICLASS OUTPUT
# =====================================================

if len(shap_values.values.shape) == 3:

    print("\nMulticlass SHAP detected")

    print(
        f"Classes: {shap_values.values.shape[2]}"
    )

    # Use Class 0 explanation

    shap_plot_values = shap_values[:, :, 0]

else:

    shap_plot_values = shap_values

# =====================================================
# SHAP BEESWARM
# =====================================================

print("\nGenerating Beeswarm Plot...")

shap.plots.beeswarm(
    shap_plot_values,
    max_display=20,
    show=False
)

plt.savefig(
    "outputs/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SHAP BAR PLOT
# =====================================================

print("\nGenerating Bar Plot...")

shap.plots.bar(
    shap_plot_values,
    max_display=20,
    show=False
)

plt.savefig(
    "outputs/figures/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved Files:")

print("outputs/figures/shap_summary.png")

print("outputs/figures/shap_bar.png")