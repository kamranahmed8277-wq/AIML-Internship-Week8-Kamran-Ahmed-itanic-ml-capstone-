"""
Step 16: Professional 6-Chart Capstone Dashboard
Generates reports/capstone_dashboard.png and logs it to MLflow.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow

# ============================================================
# CONFIGURATION
# ============================================================
COLOR_PRIMARY = "#7B2D8B"  # Purple
COLOR_ACCENT = "#C8960C"  # Gold
COLOR_TEXT = "#374151"  # Dark gray
DPI = 150
FIGURE_SIZE = (20, 13)
OUTPUT_PATH = "reports/capstone_dashboard.png"
RANDOM_STATE = 42

os.makedirs("reports", exist_ok=True)

# ============================================================
# 1. LOAD DATA & APPLY FEATURE ENGINEERING (must match training)
# ============================================================
print("Loading data and applying feature engineering...")

df = pd.read_csv("Dataset/train.csv")


# --- Feature Engineering (exactly as done during training) ---
# 1. Title extraction
def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title


df['Title'] = df['Name'].apply(extract_title)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Rare', 'Countess': 'Rare', 'Ms': 'Rare', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Rare',
    'Capt': 'Rare', 'Sir': 'Rare'
}
df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

# 2. Family size and IsAlone
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)


# 3. Deck from Cabin
def get_deck(cabin):
    if pd.isna(cabin):
        return 'Unknown'
    return cabin[0]


df['Deck'] = df['Cabin'].apply(get_deck)

# 4. Age bands (if used; include if your pipeline expects it)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# 5. Fare per person
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# 6. Impute missing values (if not already done in training)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop columns that are NOT used as features (including target)
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X = df.drop(columns=['Survived'] + drop_cols, errors='ignore')
y = df['Survived']

# Split into train/validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f" X_val shape: {X_val.shape}")

# ============================================================
# 2. LOAD THE CHAMPION PIPELINE
# ============================================================
print(" Loading trained pipeline from models/titanic_pipeline.pkl ...")
with open("models/titanic_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Get predictions and probabilities
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

print(" Predictions generated.")

# ============================================================
# 3. PREPARE PREPROCESSED DATA FOR COMPARISON MODELS
# ============================================================
# Extract the preprocessor from the pipeline
preprocessor = pipeline.named_steps.get('preprocessor', None)
if preprocessor is not None:
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
else:
    # If pipeline doesn't have a separate preprocessor, we need to handle manually
    # but usually it does. Fallback: use the whole pipeline minus the classifier.
    # That is complex; we'll simply use X_train, X_val as is (assuming they are numeric)
    X_train_proc = X_train.values
    X_val_proc = X_val.values

# ============================================================
# 4. TRAIN COMPARISON MODELS (Logistic, RF, XGBoost)
# ============================================================
print(" Training comparison models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             use_label_encoder=False, eval_metric='logloss')
}

metrics_dict = {}
for name, model in models.items():
    model.fit(X_train_proc, y_train)
    y_pred_cv = model.predict(X_val_proc)
    y_proba_cv = model.predict_proba(X_val_proc)[:, 1]

    metrics_dict[name] = {
        "Accuracy": accuracy_score(y_val, y_pred_cv),
        "Macro F1": f1_score(y_val, y_pred_cv, average='macro'),
        "ROC-AUC": auc(roc_curve(y_val, y_proba_cv)[0], roc_curve(y_val, y_proba_cv)[1])
    }

# Add the champion (pipeline) to the comparison
metrics_dict["Champion (Pipeline)"] = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Macro F1": f1_score(y_val, y_pred, average='macro'),
    "ROC-AUC": auc(roc_curve(y_val, y_proba)[0], roc_curve(y_val, y_proba)[1])
}

# ============================================================
# 5. EXTRACT FEATURE NAMES & IMPORTANCES
# ============================================================
classifier = pipeline.named_steps.get('classifier', None)
feature_importances = None
feature_names = None

if classifier is not None:
    # Try to get feature importances (for tree-based) or coefficients
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # For linear models, use absolute coefficients as importance
        feature_importances = np.abs(classifier.coef_[0])
    else:
        feature_importances = None

    # Try to get feature names from preprocessor
    if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            pass

    # If we have importances but no names, create generic names
    if feature_importances is not None:
        if feature_names is None:
            # If the preprocessor output shape matches, use generic names
            if len(feature_importances) == X_val_proc.shape[1]:
                feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]
            else:
                # Fallback: use raw column names if they exist
                feature_names = X_val.columns.tolist()
else:
    # If no classifier found, we cannot plot feature importance
    feature_importances = None

# ============================================================
# 6. CREATE THE 3x2 DASHBOARD
# ============================================================
print(" Generating 6-chart dashboard...")

fig = plt.figure(figsize=FIGURE_SIZE)
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

# ---- Plot 1: ROC Curve (Top Left) ----
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color=COLOR_ACCENT, lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', lw=1)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', color=COLOR_TEXT)
ax1.set_ylabel('True Positive Rate', color=COLOR_TEXT)
ax1.set_title('1. ROC Curve', fontsize=14, fontweight='bold', color=COLOR_TEXT)
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# ---- Plot 2: Precision-Recall Curve (Top Right) ----
ax2 = fig.add_subplot(gs[0, 1])
precision, recall, _ = precision_recall_curve(y_val, y_proba)
avg_precision = average_precision_score(y_val, y_proba)

ax2.plot(recall, precision, color=COLOR_PRIMARY, lw=2.5, label=f'AP = {avg_precision:.3f}')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall', color=COLOR_TEXT)
ax2.set_ylabel('Precision', color=COLOR_TEXT)
ax2.set_title('2. Precision-Recall Curve', fontsize=14, fontweight='bold', color=COLOR_TEXT)
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

# ---- Plot 3: Feature Importance (Top 15) (Middle Left) ----
ax3 = fig.add_subplot(gs[1, 0])
if feature_importances is not None:
    # Ensure we have matching number of names
    if len(feature_importances) != len(feature_names):
        # If mismatch, use generic names
        feature_names = [f"F{i}" for i in range(len(feature_importances))]
    # Sort by importance
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_n = min(15, len(feature_importances))
    top_idx = sorted_idx[:top_n]
    top_importances = feature_importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    ax3.barh(top_names, top_importances, color=COLOR_PRIMARY)
    ax3.set_xlabel('Importance', color=COLOR_TEXT)
    ax3.set_title('3. Feature Importance (Top 15)', fontsize=14, fontweight='bold', color=COLOR_TEXT)
    ax3.invert_yaxis()
else:
    ax3.text(0.5, 0.5, 'Feature importances not available\nfor this model type.',
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('3. Feature Importance (Top 15)', fontsize=14, fontweight='bold', color=COLOR_TEXT)

# ---- Plot 4: Confusion Matrix (Normalised) (Middle Right) ----
ax4 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_val, y_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'],
            cbar_kws={'label': 'Proportion'}, ax=ax4)
ax4.set_xlabel('Predicted', color=COLOR_TEXT)
ax4.set_ylabel('True', color=COLOR_TEXT)
ax4.set_title('4. Confusion Matrix (Normalised by True)', fontsize=14, fontweight='bold', color=COLOR_TEXT)

# ---- Plot 5: Model Comparison Heatmap (Bottom Left) ----
ax5 = fig.add_subplot(gs[2, 0])
df_metrics = pd.DataFrame(metrics_dict).T
# Reorder so champion appears first (or last)
df_metrics = df_metrics.reindex(['Champion (Pipeline)'] +
                                [m for m in df_metrics.index if m != 'Champion (Pipeline)'])
sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='YlGnBu',
            cbar_kws={'label': 'Score'}, ax=ax5)
ax5.set_title('5. Model Comparison (3 Models × 3 Metrics)', fontsize=14, fontweight='bold', color=COLOR_TEXT)

# ---- Plot 6: Demographic Subgroup F1 (Sex × Pclass) (Bottom Right) ----
ax6 = fig.add_subplot(gs[2, 1])
# We need the original validation data with Sex and Pclass
val_df = X_val.copy()
val_df['Survived_True'] = y_val
val_df['Predicted'] = y_pred

# Group by Sex and Pclass
subgroup_data = []
for sex in val_df['Sex'].unique():
    for pclass in sorted(val_df['Pclass'].unique()):
        mask = (val_df['Sex'] == sex) & (val_df['Pclass'] == pclass)
        if mask.sum() > 0:
            f1 = f1_score(val_df.loc[mask, 'Survived_True'], val_df.loc[mask, 'Predicted'])
            subgroup_data.append({'Sex': sex, 'Pclass': pclass, 'F1': f1, 'Count': mask.sum()})

df_subgroup = pd.DataFrame(subgroup_data)
pivot = df_subgroup.pivot(index='Sex', columns='Pclass', values='F1')
pivot.plot(kind='bar', ax=ax6, color=[COLOR_PRIMARY, COLOR_ACCENT, '#9CA3AF'], edgecolor='black')
ax6.set_ylim([0, 1])
ax6.set_ylabel('F1 Score', color=COLOR_TEXT)
ax6.set_xlabel('Sex', color=COLOR_TEXT)
ax6.set_title('6. Demographic Subgroup F1 (Sex × Pclass)', fontsize=14, fontweight='bold', color=COLOR_TEXT)
ax6.legend(title='Pclass', labels=['1st', '2nd', '3rd'])
ax6.grid(True, alpha=0.3, axis='y')
ax6.tick_params(axis='x', rotation=0)

# ---- Main Title ----
fig.suptitle('Capstone Evaluation Dashboard — Titanic Survival Prediction',
             fontsize=20, fontweight='bold', color=COLOR_TEXT, y=0.98)

# ---- Save ----
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
plt.close()
print(f" Dashboard saved to {OUTPUT_PATH}")

# ============================================================
# 7. LOG TO MLflow
# ============================================================
print(" Logging dashboard to MLflow...")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-capstone")

with mlflow.start_run(run_name="capstone_dashboard"):
    mlflow.log_artifact(OUTPUT_PATH)
    # Log champion metrics
    for metric, value in metrics_dict["Champion (Pipeline)"].items():
        mlflow.log_metric(f"champion_{metric.lower().replace(' ', '_')}", value)
    print("Dashboard logged as MLflow artifact.")

print("\n🎉 Step 16 Complete! Check reports/capstone_dashboard.png and MLflow UI.")