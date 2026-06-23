# ============================================================
# Titanic Machine Learning Capstone Project
# Step 3: Feature Engineering
# ============================================================

# Objectives:
# 1. Extract and map titles from Name.
# 2. Create FamilySize and IsAlone.
# 3. Create AgeGroup (binned).
# 4. Extract Deck from Cabin (fill unknown with 'U').
# 5. Drop unused columns (Name, Ticket, Cabin, and PassengerId from train only).
# 6. Print final feature lists.

# ============================================================
# Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import re

# ============================================================
# Load Datasets (if not already loaded in previous steps)
# ============================================================

train_path = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\train.csv"
test_path  = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\test.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# ============================================================
# 1. Extract Title from Name
# ============================================================

print("\n[1] Extracting titles from Name...")

# Extract title using regex: pattern captures word before a dot
df_train['Title'] = df_train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_test['Title']  = df_test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Map titles to standard categories
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Ms': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Capt': 'Rare',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Lady': 'Rare',
    'Sir': 'Rare',
    'Jonkheer': 'Rare',
    'Countess': 'Rare'
}

df_train['Title'] = df_train['Title'].map(title_mapping).fillna('Rare')
df_test['Title']  = df_test['Title'].map(title_mapping).fillna('Rare')

print("Title mapping completed.")
print("Training set title distribution:")
print(df_train['Title'].value_counts())

# ============================================================
# 2. Family Size and IsAlone
# ============================================================

print("\n[2] Creating FamilySize and IsAlone...")

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize']  = df_test['SibSp'] + df_test['Parch'] + 1

df_train['IsAlone'] = (df_train['FamilySize'] == 1).astype(int)
df_test['IsAlone']  = (df_test['FamilySize'] == 1).astype(int)

print("FamilySize created (includes self).")
print("IsAlone: 1 if alone, else 0.")
print("Training set - IsAlone counts:")
print(df_train['IsAlone'].value_counts())

# ============================================================
# 3. Age Group (binned)
# ============================================================

print("\n[3] Creating AgeGroup bins...")

bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']

# We'll impute missing Age later, but for now we can bin existing ages.
# For consistency, we create the bins even with NaN; they will be filled later in preprocessing.
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
df_test['AgeGroup']  = pd.cut(df_test['Age'], bins=bins, labels=labels, right=False)

print("AgeGroup created with bins:", bins)
print("Labels:", labels)
print("Training set AgeGroup distribution:")
print(df_train['AgeGroup'].value_counts(dropna=False))

# ============================================================
# 4. Extract Deck from Cabin
# ============================================================

print("\n[4] Extracting Deck from Cabin...")

# Cabin is string; take first character
df_train['Deck'] = df_train['Cabin'].str[0]
df_test['Deck']  = df_test['Cabin'].str[0]

# Replace NaN with 'U' for Unknown
df_train['Deck'] = df_train['Deck'].fillna('U')
df_test['Deck']  = df_test['Deck'].fillna('U')

print("Deck extracted and 'U' used for unknown.")
print("Training set Deck distribution:")
print(df_train['Deck'].value_counts())

# ============================================================
# 5. Drop unused columns
# ============================================================

print("\n[5] Dropping columns not used in modelling...")

# Columns to drop from both sets (common)
drop_cols = ['Name', 'Ticket', 'Cabin']

# For training set, also drop PassengerId (since it's not a feature)
df_train.drop(columns=drop_cols + ['PassengerId'], inplace=True, errors='ignore')

# For test set, keep PassengerId for submission, drop the others
df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

print("Dropped: Name, Ticket, Cabin (both sets); PassengerId removed from training only.")
print("Train set columns after drop:")
print(df_train.columns.tolist())
print("\nTest set columns after drop:")
print(df_test.columns.tolist())

# ============================================================
# 6. Final feature verification
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE LISTS")
print("=" * 60)

print("\nTraining set features (excluding target 'Survived'):")
train_features = [col for col in df_train.columns if col != 'Survived']
print(train_features)

print("\nTest set features:")
test_features = df_test.columns.tolist()
print(test_features)

print("\nCheck: Are both sets having the same features (excluding target)?")
# For modelling, we want same set of features in both (except target not in test)
train_set = set(train_features)
test_set = set(test_features)
if train_set == test_set:
    print("Both sets have identical feature sets.")
else:
    print("Mismatch! Check differences:")
    print("Only in train:", train_set - test_set)
    print("Only in test:", test_set - train_set)

print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 60)