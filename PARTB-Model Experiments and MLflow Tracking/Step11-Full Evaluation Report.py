# ============================================================
# Titanic Machine Learning Capstone Project
# Step 11: Full Evaluation Report (with bias audit)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn

# ============================================================
# 1. Set tracking URI and locate champion model
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
client = MlflowClient()

print("=" * 60)
print("STEP 11: FULL EVALUATION REPORT")
print("=" * 60)

# Try to load from Model Registry first
model_name = "titanic_survival"
pipeline = None
run_id = None

try:
    # Get the latest Production version
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f" Loading champion from registry (version {prod_version.version})")
        pipeline = mlflow.sklearn.load_model(model_uri)
    else:
        raise ValueError("No Production version found.")
except Exception as e:
    print(f" Registry load failed: {e}")
    print("Falling back to best run from all experiments...")
    # Query all runs and pick the best by ROC-AUC
    all_runs = mlflow.search_runs()
    if all_runs.empty:
        raise ValueError(" No runs found in any experiment.")
    # Filter top-level
    if 'tags.mlflow.parentRunId' in all_runs.columns:
        top = all_runs[all_runs['tags.mlflow.parentRunId'].isna()]
    else:
        top = all_runs
    # Keep only runs with ROC-AUC
    top = top.dropna(subset=['metrics.roc_auc'])
    if top.empty:
        raise ValueError(" No runs with ROC-AUC metric.")
    best = top.sort_values('metrics.roc_auc', ascending=False).iloc[0]
    run_id = best['run_id']
    # Find model artifact
    def find_model_artifact(run_id):
        candidates = ['baseline_model', 'random_forest_model', 'gradient_boosting_model', 'GradientBoosting_Best']
        for path in candidates:
            try:
                client.list_artifacts(run_id, path=path)
                return path
            except:
                continue
        for art in client.list_artifacts(run_id):
            if art.is_dir:
                sub = client.list_artifacts(run_id, path=art.path)
                for s in sub:
                    if 'model.pkl' in s.path or 'model' in s.path:
                        return art.path
        return None
    model_path = find_model_artifact(run_id)
    if model_path is None:
        raise ValueError(" Could not find model artifact.")
    model_uri = f"runs:/{run_id}/{model_path}"
    print(f" Loading best run: {best['run_name'] if 'run_name' in best else best['run_id']} (ROC-AUC = {best['metrics.roc_auc']:.4f})")
    pipeline = mlflow.sklearn.load_model(model_uri)

print(" Model loaded successfully.\n")

# ============================================================
# 2. Load and engineer data (same as before)
# ============================================================

def engineer_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Ms': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Capt': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Countess': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')
    return df

train_path = os.path.join(PROJECT_ROOT, "Dataset", "train.csv")
df_original = pd.read_csv(train_path)
df_engineered = engineer_features(df_original)

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'Deck', 'IsAlone']
X = df_engineered.drop('Survived', axis=1)
y = df_engineered['Survived']

# Use the same train/validation split as in training (random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# ============================================================
# 3. Predict on validation set
# ============================================================

y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

# ============================================================
# 4. Full classification report
# ============================================================

report = classification_report(y_val, y_pred, target_names=['Died', 'Survived'])
print("\n CLASSIFICATION REPORT:")
print(report)

# ============================================================
# 5. 6‑panel evaluation figure
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Champion Model Evaluation Report', fontsize=20)

# (1) Confusion matrix (counts)
cm = confusion_matrix(y_val, y_pred)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
disp1.plot(ax=axes[0,0], cmap='Blues', values_format='d')
axes[0,0].set_title('Confusion Matrix (Counts)')

# (2) Normalised confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Died', 'Survived'])
disp2.plot(ax=axes[0,1], cmap='Blues', values_format='.2f')
axes[0,1].set_title('Confusion Matrix (Normalized)')

# (3) ROC curve with AUC
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)
axes[0,2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0,2].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axes[0,2].set_xlim([0.0, 1.0])
axes[0,2].set_ylim([0.0, 1.05])
axes[0,2].set_xlabel('False Positive Rate')
axes[0,2].set_ylabel('True Positive Rate')
axes[0,2].set_title('ROC Curve')
axes[0,2].legend(loc="lower right")

# (4) Precision‑Recall curve
precision, recall, _ = precision_recall_curve(y_val, y_proba)
avg_precision = average_precision_score(y_val, y_proba)
axes[1,0].plot(recall, precision, color='green', lw=2, label=f'PR (AP = {avg_precision:.3f})')
axes[1,0].set_xlabel('Recall')
axes[1,0].set_ylabel('Precision')
axes[1,0].set_title('Precision-Recall Curve')
axes[1,0].legend(loc="lower left")

# (5) Calibration plot (reliability diagram)
prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
axes[1,1].plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
axes[1,1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
axes[1,1].set_xlabel('Mean Predicted Probability')
axes[1,1].set_ylabel('Fraction of Positives')
axes[1,1].set_title('Calibration Plot')
axes[1,1].legend()

# (6) Learning curve (F1 vs training set size)
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
axes[1,2].plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training F1')
axes[1,2].plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross-validation F1')
axes[1,2].set_xlabel('Training Examples')
axes[1,2].set_ylabel('F1 (macro)')
axes[1,2].set_title('Learning Curve')
axes[1,2].legend(loc='best')
axes[1,2].grid(True)

plt.tight_layout()

os.makedirs('reports', exist_ok=True)
fig_path = 'reports/evaluation_report.png'
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"\nEvaluation figure saved to {fig_path}")

# ============================================================
# 6. Fairness / Bias Audit (by Sex and Pclass)
# ============================================================

eval_df = X_val.copy()
eval_df['y_true'] = y_val
eval_df['y_pred'] = y_pred

fairness_metrics = []

# By Sex
for sex in eval_df['Sex'].unique():
    sub = eval_df[eval_df['Sex'] == sex]
    acc = accuracy_score(sub['y_true'], sub['y_pred'])
    f1 = f1_score(sub['y_true'], sub['y_pred'])
    fairness_metrics.append({
        'Group': f'Sex_{sex}',
        'Accuracy': acc,
        'F1': f1,
        'Count': len(sub)
    })

# By Pclass
for pclass in eval_df['Pclass'].unique():
    sub = eval_df[eval_df['Pclass'] == pclass]
    acc = accuracy_score(sub['y_true'], sub['y_pred'])
    f1 = f1_score(sub['y_true'], sub['y_pred'])
    fairness_metrics.append({
        'Group': f'Pclass_{pclass}',
        'Accuracy': acc,
        'F1': f1,
        'Count': len(sub)
    })

fairness_df = pd.DataFrame(fairness_metrics)
print("\n⚖️ FAIRNESS METRICS (Bias Audit):")
print(fairness_df.to_string(index=False))

# ============================================================
# 7. Log everything to MLflow under the champion run
# ============================================================

if run_id:
    print("\n Logging evaluation artifacts to MLflow...")
    with mlflow.start_run(run_id=run_id, nested=False):
        # Figure
        mlflow.log_artifact(fig_path, artifact_path='evaluation')
        # Classification report
        with open('reports/classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('reports/classification_report.txt', artifact_path='evaluation')
        # Fairness metrics CSV
        fairness_df.to_csv('reports/fairness_metrics.csv', index=False)
        mlflow.log_artifact('reports/fairness_metrics.csv', artifact_path='evaluation')
        # Additional metrics (overwrite if already exist)
        mlflow.log_metrics({
            'val_accuracy': accuracy_score(y_val, y_pred),
            'val_f1_macro': f1_score(y_val, y_pred, average='macro'),
            'val_roc_auc': roc_auc_score(y_val, y_proba),
            'val_precision': precision_score(y_val, y_pred),
            'val_recall': recall_score(y_val, y_pred)
        })
        print(" Evaluation logged to MLflow under the champion run.")
else:
    print(" No run ID found – skipping MLflow logging.")

print("\n" + "=" * 60)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 60)