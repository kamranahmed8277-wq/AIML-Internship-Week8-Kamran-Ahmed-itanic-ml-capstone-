# ============================================================
# Titanic Machine Learning Capstone Project
# Step 5: Baseline Model and First MLflow Run (FIXED)
# ============================================================

# Objectives:
# 1. Build a pipeline: ColumnTransformer (preprocessing) + LogisticRegression.
# 2. Train on X_train, evaluate on X_val.
# 3. Log parameters, metrics, confusion matrix PNG, and model to MLflow.
# 4. Print classification report.

# ============================================================
# Import Libraries
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import mlflow
import mlflow.sklearn

# ============================================================
# Project root and tracking URI (consistent with Step 10)
# ============================================================

# This script is inside: .../PARTB-Model Experiments and MLflow Tracking/
# So the project root is the parent directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================================
# Load and engineer features (reusing Step 3 logic)
# ============================================================

def engineer_features(df):
    """Apply feature engineering to Titanic dataset."""
    df = df.copy()
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Ms': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Capt': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Countess': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age group
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Deck
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    # Drop unused columns
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')

    return df

# Load dataset from the 'Dataset' folder in the project root
train_path = os.path.join(PROJECT_ROOT, "Dataset", "train.csv")
df_original = pd.read_csv(train_path)
df_engineered = engineer_features(df_original)

# ============================================================
# Define features and target
# ============================================================

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'Deck', 'IsAlone']

X = df_engineered.drop('Survived', axis=1)
y = df_engineered['Survived']

# ============================================================
# Train/validation split (stratified)
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 5: BASELINE MODEL TRAINING")
print("=" * 60)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ============================================================
# Build preprocessing pipeline
# ============================================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ============================================================
# Build full pipeline with LogisticRegression
# ============================================================

logreg = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logreg)
])

# ============================================================
# Train and evaluate
# ============================================================

print("\n[1] Training logistic regression baseline...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average='macro')
roc_auc = roc_auc_score(y_val, y_proba)

print("\n[2] Validation performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n[3] Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))

# ============================================================
# Confusion Matrix plot (saved to project root /reports)
# ============================================================

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Logistic Regression Baseline')
plt.tight_layout()

# Save to reports folder in project root
reports_dir = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(reports_dir, exist_ok=True)
cm_path = os.path.join(reports_dir, "confusion_matrix_baseline.png")
plt.savefig(cm_path, dpi=120)
plt.close()

# ============================================================
# MLflow Logging
# ============================================================

print("\n[4] Logging run to MLflow...")

with mlflow.start_run(run_name="LogReg_Baseline"):

    # Parameters
    mlflow.log_params({
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "random_state": 42
    })

    # Metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    })

    # Confusion Matrix Artifact
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

    # Save model using cloudpickle (avoids skops serialisation issues)
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="baseline_model",
        serialization_format="cloudpickle"
    )

    print("✓ Run logged successfully")
    print("Run ID:", mlflow.active_run().info.run_id)

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 60)