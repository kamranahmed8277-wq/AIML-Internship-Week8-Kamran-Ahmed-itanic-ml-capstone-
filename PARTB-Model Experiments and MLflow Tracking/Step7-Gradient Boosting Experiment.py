# ============================================================
# Titanic Machine Learning Capstone Project
# Step 7: Gradient Boosting Experiment
# ============================================================

# Objectives:
# 1. Build a pipeline: Preprocessor + GradientBoostingClassifier.
# 2. Train on X_train, evaluate on X_val.
# 3. Log hyperparameters, metrics, training time, and deviance plot to MLflow.
# 4. Compare ROC‑AUC of Logistic Regression, Random Forest, and Gradient Boosting.

# ============================================================
# Import Libraries
# ============================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.sklearn

# ============================================================
# Set MLflow tracking URI (SQLite)
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================================
# Load and engineer features (same as Step 5/6)
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


train_path = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\train.csv"
df_original = pd.read_csv(train_path)
df_engineered = engineer_features(df_original)

# ============================================================
# Define features and target
# ============================================================

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'Deck', 'IsAlone']

X = df_engineered.drop('Survived', axis=1)
y = df_engineered['Survived']

# Train/validation split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 7: GRADIENT BOOSTING EXPERIMENT")
print("=" * 60)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ============================================================
# Build preprocessing pipeline (same as before)
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
# Train Logistic Regression (baseline) for comparison
# ============================================================

print("\n[1] Training baseline (Logistic Regression)...")
logreg = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logreg)
])
baseline_pipeline.fit(X_train, y_train)
y_proba_baseline = baseline_pipeline.predict_proba(X_val)[:, 1]
baseline_roc_auc = roc_auc_score(y_val, y_proba_baseline)
baseline_accuracy = accuracy_score(y_val, baseline_pipeline.predict(X_val))
baseline_f1 = f1_score(y_val, baseline_pipeline.predict(X_val), average='macro')
print(f"Baseline ROC-AUC: {baseline_roc_auc:.4f}")

# ============================================================
# Train Random Forest (from Step 6) for comparison
# ============================================================

print("\n[2] Training Random Forest (for comparison)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])
rf_pipeline.fit(X_train, y_train)
y_proba_rf = rf_pipeline.predict_proba(X_val)[:, 1]
rf_roc_auc = roc_auc_score(y_val, y_proba_rf)
rf_accuracy = accuracy_score(y_val, rf_pipeline.predict(X_val))
rf_f1 = f1_score(y_val, rf_pipeline.predict(X_val), average='macro')
print(f"Random Forest ROC-AUC: {rf_roc_auc:.4f}")

# ============================================================
# Train Gradient Boosting
# ============================================================

print("\n[3] Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb)
])

# Measure training time
start_time = time.time()
gb_pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred_gb = gb_pipeline.predict(X_val)
y_proba_gb = gb_pipeline.predict_proba(X_val)[:, 1]

# Evaluation metrics
gb_accuracy = accuracy_score(y_val, y_pred_gb)
gb_precision = precision_score(y_val, y_pred_gb)
gb_recall = recall_score(y_val, y_pred_gb)
gb_f1_macro = f1_score(y_val, y_pred_gb, average='macro')
gb_roc_auc = roc_auc_score(y_val, y_proba_gb)

print(f"Gradient Boosting metrics:")
print(f"  Accuracy:  {gb_accuracy:.4f}")
print(f"  Precision: {gb_precision:.4f}")
print(f"  Recall:    {gb_recall:.4f}")
print(f"  F1 (macro): {gb_f1_macro:.4f}")
print(f"  ROC-AUC:   {gb_roc_auc:.4f}")
print(f"  Training time: {train_time:.2f} seconds")

print("\n[4] Classification Report:")
print(classification_report(y_val, y_pred_gb, target_names=['Died', 'Survived']))

# ============================================================
# Staged Deviance Plot (training loss per iteration)
# ============================================================

print("\n[5] Generating staged deviance plot...")
# Retrieve staged deviance (training loss)
deviance = gb_pipeline.named_steps['classifier'].train_score_

plt.figure(figsize=(8, 5))
plt.plot(deviance, color='darkorange')
plt.xlabel('Boosting Iteration')
plt.ylabel('Deviance (Training Loss)')
plt.title('Staged Deviance - Gradient Boosting')
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('reports', exist_ok=True)
deviance_path = 'reports/deviance_plot_gb.png'
plt.savefig(deviance_path, dpi=120)
plt.close()
print(f"Deviance plot saved to {deviance_path}")

# ============================================================
# ROC‑AUC Comparison of All Three Models
# ============================================================

print("\n[6] ROC‑AUC Comparison of All Models")
print("-" * 40)
print(f"{'Model':<20} {'ROC-AUC':<10}")
print("-" * 40)
print(f"{'Logistic Regression':<20} {baseline_roc_auc:.4f}")
print(f"{'Random Forest':<20} {rf_roc_auc:.4f}")
print(f"{'Gradient Boosting':<20} {gb_roc_auc:.4f}")
print("-" * 40)

# Determine which is best
best_model = max(
    [('Logistic Regression', baseline_roc_auc),
     ('Random Forest', rf_roc_auc),
     ('Gradient Boosting', gb_roc_auc)],
    key=lambda x: x[1]
)
print(f" Best model by ROC-AUC: {best_model[0]} ({best_model[1]:.4f})")

# ============================================================
# MLflow Logging for Gradient Boosting
# ============================================================

print("\n[7] Logging Gradient Boosting run to MLflow...")

with mlflow.start_run(run_name="GradientBoosting_v1"):
    # Log hyperparameters
    mlflow.log_params({
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'random_state': 42
    })

    # Log evaluation metrics
    mlflow.log_metrics({
        'accuracy': gb_accuracy,
        'precision': gb_precision,
        'recall': gb_recall,
        'f1_macro': gb_f1_macro,
        'roc_auc': gb_roc_auc
    })

    # Log training time
    mlflow.log_metric('train_time_s', train_time)

    # Log deviance plot as artifact
    mlflow.log_artifact(deviance_path, artifact_path='deviance_plot')

    # Log the fitted pipeline
    mlflow.sklearn.log_model(
        sk_model=gb_pipeline,
        artifact_path='gradient_boosting_model',
        serialization_format='cloudpickle'
    )

    # Optionally log comparison table as a text artifact
    comparison_text = (
        f"Model Comparison (ROC-AUC):\n"
        f"Logistic Regression: {baseline_roc_auc:.4f}\n"
        f"Random Forest: {rf_roc_auc:.4f}\n"
        f"Gradient Boosting: {gb_roc_auc:.4f}\n"
        f"Best: {best_model[0]} ({best_model[1]:.4f})"
    )
    with open('reports/model_comparison.txt', 'w') as f:
        f.write(comparison_text)
    mlflow.log_artifact('reports/model_comparison.txt', artifact_path='comparison')

    print(f"✓ Run logged successfully")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\n" + "=" * 60)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 60)