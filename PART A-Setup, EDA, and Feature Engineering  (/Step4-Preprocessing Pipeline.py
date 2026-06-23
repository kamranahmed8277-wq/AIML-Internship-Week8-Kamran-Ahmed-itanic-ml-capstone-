# ============================================================
# Titanic Machine Learning Capstone Project
# Step 4: Preprocessing Pipeline & Data Splitting
# ============================================================

# Objectives:
# 1. Define numeric and categorical feature lists.
# 2. Build a ColumnTransformer with:
#    - Numeric: SimpleImputer(median) + StandardScaler
#    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder
# 3. Perform stratified train/validation split (80/20).
# 4. Print class balance and dataset shapes.

# ============================================================
# Import Libraries
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ============================================================
# Load the feature-engineered training set (from Step 3)
# ============================================================

# If you haven't saved it, we reload from original and re-apply engineering.
# For consistency, we'll load the original train.csv and reapply the same steps.
# (Alternatively, you could save the engineered data to a file.)

train_path = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\train.csv"
df_original = pd.read_csv(train_path)


# Re-apply feature engineering (same as Step 3)
def engineer_features(df):
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


df_train = engineer_features(df_original.copy())

print("=" * 60)
print("STEP 4: PREPROCESSING PIPELINE")
print("=" * 60)

# ============================================================
# 1. Define feature lists
# ============================================================

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'Deck', 'IsAlone']

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ============================================================
# 2. Build ColumnTransformer
# ============================================================

print("\n[1] Building preprocessing pipeline...")

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

print("ColumnTransformer created successfully.")

# ============================================================
# 3. Prepare X and y
# ============================================================

X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

print(f"\nFull dataset: X shape = {X.shape}, y shape = {y.shape}")

# ============================================================
# 4. Stratified train/validation split (80/20)
# ============================================================

print("\n[2] Performing stratified split...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
print(f"Validation set: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")

# ============================================================
# 5. Print class balance
# ============================================================

print("\n[3] Class balance (Survival):")
print("Full dataset:")
print(y.value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

print("\nTraining set:")
print(y_train.value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

print("\nValidation set:")
print(y_val.value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

# Optionally, fit the preprocessor on training data to confirm it works
# (We don't actually need to transform here; we'll do it in modelling step)
try:
    preprocessor.fit(X_train)
    print("\n Preprocessor fitted successfully on training data.")
except Exception as e:
    print(f"\n Error fitting preprocessor: {e}")

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)