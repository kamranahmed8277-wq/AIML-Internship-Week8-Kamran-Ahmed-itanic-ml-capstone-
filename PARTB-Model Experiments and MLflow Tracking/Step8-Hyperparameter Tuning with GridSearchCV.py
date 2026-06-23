# ============================================================
# Titanic Machine Learning Capstone Project
# Step 8: Hyperparameter Tuning with GridSearchCV
# ============================================================

# Objectives:
# 1. Define a parameter grid for GradientBoostingClassifier:
#    n_estimators: [200, 300]
#    learning_rate: [0.05, 0.10]
#    max_depth: [3, 4]
# 2. Perform 5‑fold stratified GridSearchCV over the pipeline.
# 3. Log each combination as a nested MLflow child run inside a parent run.
# 4. After search, log best params, final validation metrics, and save the best pipeline.

# ============================================================
# Import Libraries
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.sklearn

# ============================================================
# Set MLflow tracking URI (SQLite)
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================================
# Load and engineer features (same as previous steps)
# ============================================================

def engineer_features(df):
    """Apply feature engineering to Titanic dataset."""
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

# Train/validation split (stratified) – keep validation set for final evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 8: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
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
# Build full pipeline with GradientBoostingClassifier
# ============================================================

gb = GradientBoostingClassifier(random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb)
])

# ============================================================
# Define parameter grid
# ============================================================

param_grid = {
    'classifier__n_estimators': [200, 300],
    'classifier__learning_rate': [0.05, 0.10],
    'classifier__max_depth': [3, 4]
}

# ============================================================
# Set up GridSearchCV with stratified 5‑fold cross‑validation
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    return_train_score=False,  # we only want validation scores
    verbose=1
)

# ============================================================
# Start MLflow parent run and perform GridSearch
# ============================================================

print("\n[1] Starting GridSearchCV with 8 combinations × 5 folds = 40 fits...")

with mlflow.start_run(run_name="GradientBoosting_GridSearch") as parent_run:
    # We'll manually iterate over parameter combinations to log each as a child run
    # GridSearchCV doesn't automatically log to MLflow, so we do it ourselves.
    # We'll use GridSearchCV to fit and then extract results.

    # Fit grid search (this will run all combinations)
    grid_search.fit(X_train, y_train)

    # Extract all results
    results = grid_search.cv_results_

    # Log each combination as a child run
    for i in range(len(results['params'])):
        params = results['params'][i]
        mean_f1 = results['mean_test_score'][i]
        std_f1 = results['std_test_score'][i]

        # Create a descriptive run name
        run_name = f"GB_n{params['classifier__n_estimators']}_lr{params['classifier__learning_rate']}_d{params['classifier__max_depth']}"

        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            # Log parameters
            mlflow.log_params({
                'n_estimators': params['classifier__n_estimators'],
                'learning_rate': params['classifier__learning_rate'],
                'max_depth': params['classifier__max_depth']
            })
            # Log CV metrics
            mlflow.log_metrics({
                'cv_mean_f1_macro': mean_f1,
                'cv_std_f1_macro': std_f1
            })
            # Optionally log the rank
            mlflow.log_metric('rank', results['rank_test_score'][i])

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n[2] GridSearch complete. Best CV mean F1: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Log best parameters and score at parent level
    mlflow.log_params({k.replace('classifier__', ''): v for k, v in best_params.items()})
    mlflow.log_metric('best_cv_f1_macro', best_score)

    # ============================================================
    # Evaluate best model on validation set
    # ============================================================

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred)
    val_recall = recall_score(y_val, y_pred)
    val_f1_macro = f1_score(y_val, y_pred, average='macro')
    val_roc_auc = roc_auc_score(y_val, y_proba)

    print("\n[3] Best model validation performance:")
    print(f"  Accuracy:  {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1 (macro): {val_f1_macro:.4f}")
    print(f"  ROC-AUC:   {val_roc_auc:.4f}")

    # Log validation metrics at parent level
    mlflow.log_metrics({
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1_macro': val_f1_macro,
        'val_roc_auc': val_roc_auc
    })

    # ============================================================
    # Save and log the best pipeline
    # ============================================================

    print("\n[4] Logging best pipeline to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path='GradientBoosting_Best',
        serialization_format='cloudpickle'
    )

    # Optionally log classification report as a text artifact
    report = classification_report(y_val, y_pred, target_names=['Died', 'Survived'])
    with open('reports/best_model_classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('reports/best_model_classification_report.txt', artifact_path='reports')

    print(f"\n✓ Parent run logged successfully with ID: {parent_run.info.run_id}")
    print(f"✓ Best model saved as 'GradientBoosting_Best'")

print("\n" + "=" * 60)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 60)