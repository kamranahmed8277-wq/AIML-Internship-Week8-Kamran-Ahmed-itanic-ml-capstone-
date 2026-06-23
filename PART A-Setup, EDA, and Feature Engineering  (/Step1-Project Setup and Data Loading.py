# ============================================================
# TITANIC MACHINE LEARNING CAPSTONE PROJECT
# STEP 1: PROJECT SETUP + DATA LOADING + BASIC INSPECTION
# ============================================================

"""
STEP 1 OBJECTIVE:
------------------------------------------------------------
1. Create project folder structure
2. Load Titanic dataset (train + test)
3. Perform basic data inspection
   - shape
   - dtypes
   - head
   - missing values
4. Initialize MLflow tracking safely
------------------------------------------------------------
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import mlflow

print("\n==============================")
print("STEP 1 STARTED")
print("==============================\n")

# ============================================================
# 2. CREATE PROJECT STRUCTURE
# ============================================================

folders = [
    "data",
    "notebooks",
    "src",
    "models",
    "reports",
    "app",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✔ Project structure created successfully")

# ============================================================
# 3. LOAD DATASET
# ============================================================

train_path = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\train.csv"
test_path  = r"C:\Users\J J LAPTOP\Desktop\End-to-End Machine Learning\Dataset\test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("✔ Dataset loaded successfully")

# ============================================================
# 4. DATA OVERVIEW
# ============================================================

print("\n==============================")
print("DATA SHAPE")
print("==============================")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\n==============================")
print("DATA TYPES")
print("==============================")

print("Train dtypes:\n", df_train.dtypes)
print("\nTest dtypes:\n", df_test.dtypes)

print("\n==============================")
print("FIRST 5 ROWS (TRAIN)")
print("==============================")

print(df_train.head())

print("\n==============================")
print("FIRST 5 ROWS (TEST)")
print("==============================")

print(df_test.head())

# ============================================================
# 5. DATA INFORMATION
# ============================================================

print("\n==============================")
print("DATA INFO")
print("==============================")

df_train.info()
df_test.info()

# ============================================================
# 6. MISSING VALUES CHECK
# ============================================================

print("\n==============================")
print("MISSING VALUES (TRAIN)")
print("==============================")

print(df_train.isnull().sum())

print("\n==============================")
print("MISSING VALUES (TEST)")
print("==============================")

print(df_test.isnull().sum())

# ============================================================
# 7. MLflow INITIALIZATION (SAFE VERSION)
# ============================================================

"""
IMPORTANT FIX:
We use sqlite database to avoid Windows permission issues
"""

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "titanic-capstone"
mlflow.set_experiment(experiment_name)

print("\n==============================")
print("MLFLOW INITIALIZED")
print("==============================")

print("Tracking URI: sqlite:///mlflow.db")
print("Experiment:", experiment_name)

# ============================================================
# STEP COMPLETION MESSAGE
# ============================================================

print("\n==============================")
print("STEP 1 COMPLETED SUCCESSFULLY")
print("==============================")

print("""
Completed:
✔ Project structure created
✔ Dataset loaded
✔ Shape checked
✔ Data types checked
✔ Missing values analyzed
✔ MLflow initialized

Next Step:
➡ Step 2: Exploratory Data Analysis (EDA)
""")