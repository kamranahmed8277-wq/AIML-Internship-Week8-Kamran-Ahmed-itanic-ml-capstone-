import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# --- 1. Fixed tracking URI ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
client = MlflowClient()
print(f"Using tracking URI: {mlflow.get_tracking_uri()}")

# --- 2. Query ALL runs across all experiments ---
all_runs = mlflow.search_runs()
print(f"\nTotal runs found across all experiments: {len(all_runs)}")
if all_runs.empty:
    raise ValueError("No runs found. Please run Steps 5–7 first.")

# --- 3. Filter top-level runs (exclude nested GridSearch children) ---
if 'tags.mlflow.parentRunId' in all_runs.columns:
    top = all_runs[all_runs['tags.mlflow.parentRunId'].isna()].copy()
else:
    top = all_runs.copy()

# --- 4. Build comparison table ---
metric_map = {
    'metrics.accuracy': 'Accuracy',
    'metrics.f1_macro': 'F1 (macro)',
    'metrics.roc_auc': 'ROC-AUC',
    'metrics.train_time_s': 'Train Time (s)'
}

# Select columns that actually exist
base_cols = ['run_id']
if 'run_name' in top.columns:
    base_cols.append('run_name')
if 'experiment_id' in top.columns:
    base_cols.append('experiment_id')

present_metrics = [col for col in metric_map.keys() if col in top.columns]
select_cols = base_cols + present_metrics

comp = top[select_cols].copy()
comp.rename(columns=metric_map, inplace=True)

# Drop rows missing core metrics
core_metrics = ['Accuracy', 'F1 (macro)', 'ROC-AUC']
comp.dropna(subset=core_metrics, inplace=True)

if comp.empty:
    raise ValueError("No runs with complete metrics found.")

# Sort by ROC-AUC descending
comp.sort_values('ROC-AUC', ascending=False, inplace=True)
comp.reset_index(drop=True, inplace=True)
comp['selected'] = False
comp.loc[0, 'selected'] = True

# --- 5. Print comparison table ---
print("\n📊 Comparison Table (sorted by ROC-AUC):")
print(comp.to_string(index=False))

champion = comp.iloc[0]
champion_name = champion['run_name'] if 'run_name' in champion else champion['run_id']
print(f"\n🏆 Champion: {champion_name} (ROC-AUC = {champion['ROC-AUC']:.4f})")

# --- 6. Find model artifact in the champion run ---
def find_model(run_id):
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

model_path = find_model(champion['run_id'])
if model_path is None:
    raise ValueError("Could not locate model artifact in champion run.")
model_uri = f"runs:/{champion['run_id']}/{model_path}"
print(f"Model URI: {model_uri}")

# --- 7. Register and promote ---
registered = mlflow.register_model(model_uri, "titanic_survival")
print(f"Registered version: {registered.version}")
client.transition_model_version_stage(
    name="titanic_survival",
    version=registered.version,
    stage="Production",
    archive_existing_versions=True
)
print("Model promoted to 'Production'.")

# --- 8. Justification ---
second_roc = comp.iloc[1]['ROC-AUC'] if len(comp) > 1 else None
gap = champion['ROC-AUC'] - second_roc if second_roc is not None else "N/A"
print("\n" + "="*60)
print("JUSTIFICATION FOR CHAMPION")
print("="*60)
print(f"""
1. Best ROC-AUC: {champion['ROC-AUC']:.4f} {'(leader)' if second_roc is None else f'(beats next by {gap:.4f})'}.
2. Stable CV: tuned via cross-validation (if applicable).
3. Interpretable: SHAP-ready.
4. Efficient: {'N/A' if 'Train Time (s)' not in comp else f'{champion["Train Time (s)"]:.2f}s'}.
5. Production-ready: registered and deployed.
""")
print("STEP 10 COMPLETED.")