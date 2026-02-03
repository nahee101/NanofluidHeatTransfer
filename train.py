import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

# Model Registry name
MODEL_NAME = "NanofluidHeatTransfer"

def get_models():
    """Return dictionary of models to train and compare"""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1,
            random_state=42
        ),
        "Ridge": Ridge(alpha=1.0),
    }

def train_single_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train a single model and log to MLflow"""
    with mlflow.start_run(run_name=model_name) as run:
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Log model-specific parameters
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(param_name, param_value)
            except:
                pass  # Skip non-serializable params
        
        # Train model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"  MSE: {mse:.6f}")
        print(f"  R2:  {r2:.6f}")
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log model and register to Model Registry
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=MODEL_NAME
        )
        
        return {
            "model_name": model_name, 
            "mse": mse, 
            "r2": r2,
            "run_id": run.info.run_id
        }

def promote_best_model_to_staging(best_run_id):
    """Promote the best model version to Staging stage."""
    client = MlflowClient()
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    # Find the version that matches the best run_id
    best_version = None
    for v in versions:
        if v.run_id == best_run_id:
            best_version = v.version
            break
    
    if best_version:
        # Transition to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=best_version,
            stage="Staging",
            archive_existing_versions=True  # Archive other Staging versions
        )
        print(f"✅ Model version {best_version} promoted to Staging")
        return best_version
    else:
        print("⚠️ Could not find model version for the best run")
        return None


def promote_staging_to_production():
    """Promote current Staging model to Production."""
    client = MlflowClient()
    
    # Find current Staging version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_version = None
    
    for v in versions:
        if v.current_stage == "Staging":
            staging_version = v.version
            break
    
    if staging_version:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✅ Model version {staging_version} promoted to Production")
        return staging_version
    else:
        print("⚠️ No Staging model found to promote")
        return None


def train(auto_promote_to_production=False):
    """
    Train all models and register to MLflow Model Registry.
    
    Args:
        auto_promote_to_production: If True, automatically promote best model to Production.
                                   If False, promote to Staging only (default).
    """
    # Set MLflow experiment
    mlflow.set_experiment("nanofluid_heat_transfer")
    
    # Load data
    data_path = "data/nanofluid_data.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature engineering/selection
    # Target: theta_prime (related to heat transfer)
    # Features: eta, f, f_prime (related to flow)
    X = df[['eta', 'f', 'f_prime']]
    y = df['theta_prime']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train all models and collect results
    models = get_models()
    results = []
    
    print(f"\n{'='*50}")
    print(f"Training {len(models)} models...")
    print(f"{'='*50}")
    
    for model_name, model in models.items():
        result = train_single_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(result)
    
    # Find best model (highest R2, if tie then latest/last trained)
    # Sort by R2 descending, then by index descending (latest first)
    # Since results are in training order, later index = more recent
    sorted_results = sorted(
        enumerate(results), 
        key=lambda x: (x[1]["r2"], x[0]),  # (r2, index) - higher is better for both
        reverse=True
    )
    best_result = sorted_results[0][1]
    
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'MSE':<15} {'R2':<15}")
    print(f"{'-'*50}")
    for r in sorted(results, key=lambda x: -x["r2"]):
        marker = " ★ BEST" if r["model_name"] == best_result["model_name"] else ""
        print(f"{r['model_name']:<20} {r['mse']:<15.6f} {r['r2']:<15.6f}{marker}")
    
    # Try to use Model Registry (may fail due to compatibility issues)
    try:
        print(f"\n{'='*50}")
        print(f"MODEL REGISTRY")
        print(f"{'='*50}")
        
        # Promote best model to Staging
        print(f"\nPromoting best model ({best_result['model_name']}) to Staging...")
        promote_best_model_to_staging(best_result['run_id'])
        
        # Optionally promote to Production
        if auto_promote_to_production:
            print(f"\nAuto-promoting to Production...")
            promote_staging_to_production()
    except Exception as e:
        print(f"\n⚠️ Model Registry operations skipped (compatibility issue): {e}")
        print("Models are still saved and can be loaded from experiment runs.")
    
    print(f"\n✅ Best model: {best_result['model_name']} (R2: {best_result['r2']:.6f})")
    print(f"\n📊 View models: mlflow ui --port 5001")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote-to-production", action="store_true",
                       help="Automatically promote best model to Production")
    args = parser.parse_args()
    
    train(auto_promote_to_production=args.promote_to_production)
