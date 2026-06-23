# ============================================================
# Titanic Machine Learning Capstone Project
# Step 2: Exploratory Data Analysis (EDA)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for nicer plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# ============================================================
# Load Datasets
# ============================================================

train_path = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\train.csv"
test_path  = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\test.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# (a) Missing value counts
# ============================================================

print("\n" + "-" * 40)
print("(a) Missing Values per Column")
print("-" * 40)

missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print("\nTraining set:")
print(missing_train)

missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
print("\nTest set:")
print(missing_test)

# ============================================================
# (b) Survival rates
# ============================================================

print("\n" + "-" * 40)
print("(b) Survival Rates")
print("-" * 40)

overall_survival = df_train['Survived'].mean()
print(f"Overall survival rate: {overall_survival:.2%}")

# By Pclass
surv_by_pclass = df_train.groupby('Pclass')['Survived'].mean()
print("\nBy Pclass:")
print(surv_by_pclass.round(4).map(lambda x: f"{x:.2%}"))

# By Sex
surv_by_sex = df_train.groupby('Sex')['Survived'].mean()
print("\nBy Sex:")
print(surv_by_sex.round(4).map(lambda x: f"{x:.2%}"))

# By Embarked
surv_by_embarked = df_train.groupby('Embarked')['Survived'].mean()
print("\nBy Embarked:")
print(surv_by_embarked.round(4).map(lambda x: f"{x:.2%}"))

# ============================================================
# (c) Summary statistics for Age, Fare, SibSp, Parch
# ============================================================

print("\n" + "-" * 40)
print("(c) Summary Statistics (Age, Fare, SibSp, Parch)")
print("-" * 40)

cols = ['Age', 'Fare', 'SibSp', 'Parch']
summary = df_train[cols].describe(percentiles=[.25, .5, .75])
print(summary)

# ============================================================
# Create 6‑chart EDA figure (with fixes for deprecations)
# ============================================================

print("\n" + "-" * 40)
print("Generating EDA figure...")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Titanic EDA Overview', fontsize=20, y=0.98)

# 1. Survival count bar chart
sns.countplot(x='Survived', data=df_train, ax=axes[0,0])
axes[0,0].set_title('Survival Count', fontsize=14)
# Fixed: set ticks first then labels
axes[0,0].set_xticks([0, 1])
axes[0,0].set_xticklabels(['Died (0)', 'Survived (1)'])
for p in axes[0,0].patches:
    axes[0,0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)

# 2. Survival rate by Pclass – fixed: use errorbar=None, use color instead of palette
sns.barplot(x='Pclass', y='Survived', data=df_train, ax=axes[0,1], errorbar=None, color='steelblue')
axes[0,1].set_title('Survival Rate by Pclass', fontsize=14)
axes[0,1].set_ylabel('Survival Rate')

# 3. Survival rate by Sex – fixed similarly
sns.barplot(x='Sex', y='Survived', data=df_train, ax=axes[0,2], errorbar=None, color='lightseagreen')
axes[0,2].set_title('Survival Rate by Sex', fontsize=14)
axes[0,2].set_ylabel('Survival Rate')

# 4. Age distribution coloured by Survived
bins = np.arange(0, 85, 5)
sns.histplot(data=df_train, x='Age', hue='Survived', multiple='stack', bins=bins, ax=axes[1,0])
axes[1,0].set_title('Age Distribution by Survival', fontsize=14)
axes[1,0].set_xlabel('Age')

# 5. Fare distribution (log scale) coloured by Survived
sns.histplot(data=df_train, x='Fare', hue='Survived', multiple='stack', bins=30, ax=axes[1,1], log_scale=True)
axes[1,1].set_title('Fare Distribution (log scale) by Survival', fontsize=14)
axes[1,1].set_xlabel('Fare (log)')

# 6. Correlation heatmap of numeric features
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = df_train[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap', fontsize=14)

# Adjust layout
plt.tight_layout()

# Save figure
os.makedirs('reports', exist_ok=True)
fig.savefig('reports/eda_overview.png', dpi=120, bbox_inches='tight')
print("Figure saved to reports/eda_overview.png")

# Show plot if running interactively (optional)
# plt.show()

# ============================================================
# Markdown Summary
# ============================================================

print("\n" + "=" * 60)
print("EDA FINDINGS SUMMARY (Markdown)")
print("=" * 60)

summary_md = """
### Key EDA Findings

1. **Missing Data**  
   - `Cabin` has ~77% missing (687/891) – likely unusable without heavy imputation or engineering (e.g., deck extraction).  
   - `Age` has ~20% missing (177/891) – imputation via title/class or median will be necessary.  
   - `Embarked` has only 2 missing values – easily filled with mode ('S').

2. **Strong Survival Drivers**  
   - **Sex**: Females survived at ~74% vs. males at ~19% – the most influential predictor.  
   - **Pclass**: 1st class survived ~63%, 2nd ~47%, 3rd ~24% – clear socioeconomic gradient.  
   - **Embarked**: Passengers from 'C' (Cherbourg) had ~55% survival, vs. ~34% for 'S' and ~39% for 'Q' – may reflect class composition.

3. **Numeric Correlations**  
   - `Fare` positively correlates with `Survived` (0.26) – higher fare passengers had better odds.  
   - `Pclass` negatively correlates with `Survived` (-0.34) – higher class (lower Pclass number) means higher survival.  
   - `Age` shows weak negative correlation with survival (-0.08) – younger passengers slightly more likely to survive.  
   - `SibSp`/`Parch` have near‑zero correlation – family size may need non‑linear encoding (e.g., `IsAlone`).

These insights will guide feature engineering: we will create new features like `Title`, `FamilySize`, `Deck` (from Cabin), and `IsAlone` to capture more signal.
"""

print(summary_md)

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 60)