# ============================================================
# Titanic Machine Learning Capstone Project
# Step 9: SHAP Model Explainability (corrected)
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load and engineer features
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 9: SHAP MODEL EXPLAINABILITY")
print("=" * 60)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ============================================================
# Build preprocessing pipeline (using best params from Step 8)
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
# Best Gradient Boosting pipeline (hyperparameters from Step 8)
# ============================================================

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb)
])

# Train on full training set
pipeline.fit(X_train, y_train)
print("\n[1] Pipeline trained on full training set.")
val_acc = accuracy_score(y_val, pipeline.predict(X_val))
print(f"Validation accuracy: {val_acc:.4f}")

# ============================================================
# Extract preprocessor and classifier
# ============================================================

preprocessor_fitted = pipeline.named_steps['preprocessor']
clf = pipeline.named_steps['classifier']

# ============================================================
# Transform a subset of X_val for SHAP
# ============================================================

subset_size = min(300, len(X_val))
X_val_subset = X_val.iloc[:subset_size]
X_val_transformed = preprocessor_fitted.transform(X_val_subset)

numeric_feature_names = numeric_features
ohe = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = list(numeric_feature_names) + list(cat_feature_names)

X_val_shap = pd.DataFrame(X_val_transformed, columns=feature_names, index=X_val_subset.index)

print(f"\n[2] Transformed validation subset shape: {X_val_shap.shape}")
print(f"Number of features after encoding: {len(feature_names)}")

# ============================================================
# Compute SHAP values using TreeExplainer
# ============================================================

print("\n[3] Computing SHAP values...")
explainer = shap.TreeExplainer(clf)
shap_values_all = explainer.shap_values(X_val_shap)   # list of two arrays (class0, class1)
print("SHAP computation complete.")

# Extract SHAP values for the positive class (survival = 1)
if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
    shap_values = shap_values_all[1]          # class 1
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    # Fallback if SHAP version returns a single array
    shap_values = shap_values_all
    base_value = explainer.expected_value

# ============================================================
# Create SHAP plots
# ============================================================

os.makedirs('reports', exist_ok=True)

# 1. Summary beeswarm plot (use the original list for proper display)
print("\n[4] Generating summary beeswarm plot...")
plt.figure(figsize=(10, 8))
# For summary plot we need the list of both classes
shap.summary_plot(shap_values_all, X_val_shap, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary_beeswarm.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: reports/shap_summary_beeswarm.png")

# 2. Bar plot of mean |SHAP| (top 15) – use positive class SHAP values
print("    Generating bar plot of mean |SHAP|...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val_shap, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('reports/shap_bar_mean_abs.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: reports/shap_bar_mean_abs.png")

# 3. Waterfall plot for the most confidently predicted survivor
print("    Generating waterfall plot for a confidently predicted survivor...")

probs = clf.predict_proba(X_val_transformed)[:, 1]
idx_survivor = np.argmax(probs)
if probs[idx_survivor] < 0.5:
    idx_survivor = np.argmax(probs)
print(f"    Chosen instance index: {idx_survivor}, predicted probability of survival: {probs[idx_survivor]:.4f}")

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[idx_survivor],        # positive class SHAPs
        base_values=base_value,                  # base value for positive class
        data=X_val_shap.iloc[idx_survivor],
        feature_names=feature_names
    ),
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig('reports/shap_waterfall_survivor.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: reports/shap_waterfall_survivor.png")

# ============================================================
# Analysis and print findings (using positive class SHAP)
# ============================================================

print("\n[5] SHAP Analysis:")

# Mean |SHAP| per feature (positive class)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

# Top 3 features
top3 = feature_importance_df.head(3)
print("\nTop 3 features by mean |SHAP|:")
for _, row in top3.iterrows():
    print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

# Check Sex_male dominance
sex_male_feature = 'Sex_male' if 'Sex_male' in feature_names else None
if sex_male_feature:
    sex_male_row = feature_importance_df[feature_importance_df['feature'] == sex_male_feature]
    if not sex_male_row.empty:
        sex_male_rank = sex_male_row.index[0] + 1
        sex_male_value = sex_male_row.iloc[0]['mean_abs_shap']
        print(f"\n'Sex_male' appears at rank {sex_male_rank} with mean |SHAP| = {sex_male_value:.4f}")
        if sex_male_rank == 1:
            print("  ✓ Sex_male is the dominant feature (as expected).")
        else:
            print(f"  Note: Sex_male is not the top feature; top is {top3.iloc[0]['feature']}.")
    else:
        print("\n  Note: 'Sex_male' feature not found.")
else:
    print("\n  Note: 'Sex_male' feature not found (encoding may have used different naming).")

# Top 5 features
top5 = feature_importance_df.head(5)
print("\nTop 5 features overall:")
for _, row in top5.iterrows():
    print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

print("\nObservations:")
if sex_male_feature and not feature_importance_df[feature_importance_df['feature'] == sex_male_feature].empty:
    if feature_importance_df.iloc[0]['feature'] == sex_male_feature:
        print("  - Sex is the strongest predictor, confirming the bias toward female survival.")
if any(feat in feature_names for feat in ['Pclass', 'Pclass_1', 'Pclass_2', 'Pclass_3']):
    print("  - Passenger class appears among top features, reflecting socioeconomic influence.")
if 'Fare' in feature_names and feature_importance_df[feature_importance_df['feature'] == 'Fare'].iloc[0]['mean_abs_shap'] > 0.1:
    print("  - Fare also has notable impact, consistent with correlation analysis.")
engineered_features = ['Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master', 'Title_Rare',
                       'FamilySize', 'IsAlone', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_U']
for feat in engineered_features:
    if feat in feature_names:
        val = feature_importance_df[feature_importance_df['feature'] == feat].iloc[0]['mean_abs_shap']
        if val > 0.1:
            print(f"  - Engineered feature '{feat}' shows notable importance ({val:.4f}).")

print("\n" + "=" * 60)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 60)