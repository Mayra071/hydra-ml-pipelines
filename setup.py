import shutil
import os
import mlflow

def purge_experiment(experiment_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    # Delete runs (soft delete)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    for run_id in runs["run_id"]:
        mlflow.delete_run(run_id)
        print(f"Soft deleted run: {run_id}")

    # Delete experiment (soft delete)
    mlflow.delete_experiment(exp.experiment_id)
    print(f"Soft deleted experiment '{experiment_name}'")

    # Purge .trash folder if using file-based store
    mlruns_dir = os.path.join(os.getcwd(), "mlruns", ".trash")
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
        print("🗑️ Purged .trash folder")

# Example usage
purge_experiment("training_pipeline", "file:///C:/Company/training_pipeline/mlruns")
