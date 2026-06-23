# save_champion_model.py
# Loads the champion model from MLflow and saves it as a pickle file.

import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient

# --- Set tracking URI to the root mlflow.db ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
client = MlflowClient()

print("Loading champion model from MLflow...")

# --- Try to get the Production model from the Registry ---
model_name = "titanic_survival"
pipeline = None

try:
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_version = prod_versions[0]
        run_id = prod_version.run_id
        model_uri = f"runs:/{run_id}/model"
        pipeline = mlflow.sklearn.load_model(model_uri)
        print(f" Loaded from registry (version {prod_version.version}, run {run_id})")
    else:
        raise ValueError("No Production version found.")
except Exception as e:
    print(f" Registry load failed: {e}")
    print("Falling back to best run by ROC-AUC...")
    # Query all runs across experiments
    all_runs = mlflow.search_runs()
    if all_runs.empty:
        raise RuntimeError(" No runs found.")
    # Filter top-level (ignore nested GridSearch runs)
    if 'tags.mlflow.parentRunId' in all_runs.columns:
        top = all_runs[all_runs['tags.mlflow.parentRunId'].isna()]
    else:
        top = all_runs
    top = top.dropna(subset=['metrics.roc_auc'])
    if top.empty:
        raise RuntimeError(" No runs with ROC-AUC metric.")
    best = top.sort_values('metrics.roc_auc', ascending=False).iloc[0]
    run_id = best['run_id']
    run_name = best['run_name'] if 'run_name' in best else run_id
    print(f"Best run: {run_name} (ROC-AUC = {best['metrics.roc_auc']:.4f})")


    # Find the model artifact path
    def find_model_artifact(run_id):
        candidates = ['baseline_model', 'random_forest_model', 'gradient_boosting_model', 'GradientBoosting_Best']
        for path in candidates:
            try:
                client.list_artifacts(run_id, path=path)
                return path
            except:
                continue
        # fallback: search all directories
        for art in client.list_artifacts(run_id):
            if art.is_dir:
                sub = client.list_artifacts(run_id, path=art.path)
                for s in sub:
                    if 'model.pkl' in s.path or 'model' in s.path:
                        return art.path
        return None


    model_path = find_model_artifact(run_id)
    if model_path is None:
        raise RuntimeError(" Could not find model artifact.")
    model_uri = f"runs:/{run_id}/{model_path}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    print(f" Loaded from run {run_id}")

# --- Save the pipeline as a pickle file in models/ ---
os.makedirs("models", exist_ok=True)  # create models folder at root
pickle_path = "models/titanic_pipeline.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Model saved to {pickle_path}")