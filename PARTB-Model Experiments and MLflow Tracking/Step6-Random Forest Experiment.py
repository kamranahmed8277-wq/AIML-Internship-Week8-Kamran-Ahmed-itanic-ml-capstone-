# ============================================================
# Titanic Machine Learning Capstone Project
# Step 6: Random Forest Experiment
# ============================================================

# Objectives:
# 1. Build a pipeline: Preprocessor + RandomForestClassifier.
# 2. Train on X_train, evaluate on X_val.
# 3. Log hyperparameters, metrics, feature importances plot, and model to MLflow.
# 4. Compare Random Forest vs. Logistic Regression baseline side‑by‑side.

# ============================================================
# Import Libraries
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.sklearn

# ============================================================
# Set MLflow tracking URI (SQLite)
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================================
# Load and engineer features (same as Step 5)
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
print("STEP 6: RANDOM FOREST EXPERIMENT")
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
y_pred_baseline = baseline_pipeline.predict(X_val)
y_proba_baseline = baseline_pipeline.predict_proba(X_val)[:, 1]

baseline_metrics = {
    'accuracy': accuracy_score(y_val, y_pred_baseline),
    'precision': precision_score(y_val, y_pred_baseline),
    'recall': recall_score(y_val, y_pred_baseline),
    'f1_macro': f1_score(y_val, y_pred_baseline, average='macro'),
    'roc_auc': roc_auc_score(y_val, y_proba_baseline)
}

print("Baseline metrics:")
for k, v in baseline_metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# Train Random Forest
# ============================================================

print("\n[2] Training Random Forest...")
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
y_pred_rf = rf_pipeline.predict(X_val)
y_proba_rf = rf_pipeline.predict_proba(X_val)[:, 1]

rf_metrics = {
    'accuracy': accuracy_score(y_val, y_pred_rf),
    'precision': precision_score(y_val, y_pred_rf),
    'recall': recall_score(y_val, y_pred_rf),
    'f1_macro': f1_score(y_val, y_pred_rf, average='macro'),
    'roc_auc': roc_auc_score(y_val, y_proba_rf)
}

print("Random Forest metrics:")
for k, v in rf_metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# Side‑by‑side comparison
# ============================================================

print("\n[3] Comparison: Random Forest vs Baseline")
print("-" * 60)
print(f"{'Metric':<12} {'Baseline':<12} {'Random Forest':<12} {'Improvement':<12}")
print("-" * 60)
for metric in baseline_metrics.keys():
    baseline_val = baseline_metrics[metric]
    rf_val = rf_metrics[metric]
    improvement = rf_val - baseline_val
    print(f"{metric:<12} {baseline_val:<12.4f} {rf_val:<12.4f} {improvement:+.4f}")

# ============================================================
# Feature Importances (top 20) – Horizontal Bar Chart
# ============================================================

print("\n[4] Generating feature importances plot...")

# Get feature names after preprocessing (on the entire training set)
preprocessor.fit(X_train)
feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_features))
)
# Get importances from the fitted Random Forest
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Create DataFrame of features and their importance
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot top 20 (or all if fewer)
top_n = min(20, len(feat_imp_df))
top_features = feat_imp_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(top_features['feature'], top_features['importance'], color='teal')
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importances - Random Forest')
plt.gca().invert_yaxis()  # highest at top
plt.tight_layout()

# Save the figure
os.makedirs('reports', exist_ok=True)
imp_path = 'reports/feature_importances_rf.png'
plt.savefig(imp_path, dpi=120)
plt.close()
print(f"Feature importances plot saved to {imp_path}")

# ============================================================
# MLflow logging for Random Forest
# ============================================================

print("\n[5] Logging Random Forest run to MLflow...")

with mlflow.start_run(run_name="RandomForest_v1"):
    # Log hyperparameters
    mlflow.log_params({
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_leaf': 3,
        'random_state': 42,
        'n_jobs': -1
    })

    # Log evaluation metrics
    mlflow.log_metrics(rf_metrics)

    # Log feature importance plot as artifact
    mlflow.log_artifact(imp_path, artifact_path='feature_importance')

    # Log the fitted pipeline
    mlflow.sklearn.log_model(
        sk_model=rf_pipeline,
        artifact_path='random_forest_model',
        serialization_format='cloudpickle'
    )

    print(f"✓ Run logged successfully")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\n" + "=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)