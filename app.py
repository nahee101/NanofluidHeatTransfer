from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info, Summary, generate_latest
from collections import deque
import time
import os
import glob
import threading
import json

app = Flask(__name__)

# ============================================================
# PROMETHEUS METRICS - Enhanced Monitoring
# ============================================================

# Request Metrics
REQUEST_COUNT = Counter(
    'webapp_request_count', 
    'Total Web App Requests', 
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'webapp_request_latency_seconds', 
    'Request latency', 
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Prediction Metrics
PREDICTION_COUNT = Counter(
    'prediction_total',
    'Total number of predictions made'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent making predictions',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
PREDICTION_VALUE = Summary(
    'prediction_value',
    'Distribution of prediction values'
)

# Model Metrics
MODEL_R2_SCORE = Gauge('model_r2_score', 'Current model R2 score')
MODEL_MSE_SCORE = Gauge('model_mse_score', 'Current model MSE score')
MODEL_INFO = Info('model', 'Current model information')

# Error Metrics
PREDICTION_ERROR = Gauge('prediction_error_rate', 'Rolling average prediction error')
ERROR_COUNT = Counter(
    'prediction_error_total',
    'Total prediction errors',
    ['error_type']
)

# Retrain Metrics
RETRAIN_COUNT = Counter(
    'retrain_total',
    'Total number of model retrains',
    ['trigger_type']  # manual, auto, scheduled
)
RETRAIN_DURATION = Histogram(
    'retrain_duration_seconds',
    'Time spent retraining model',
    buckets=[10, 30, 60, 120, 300, 600]
)
RETRAIN_IN_PROGRESS = Gauge('retrain_in_progress', 'Whether retraining is in progress')

# Data Drift Metrics
FEATURE_MEAN = Gauge('feature_mean', 'Mean value of input features', ['feature'])
FEATURE_STD = Gauge('feature_std', 'Standard deviation of input features', ['feature'])
FEATURE_MIN = Gauge('feature_min', 'Minimum value of input features', ['feature'])
FEATURE_MAX = Gauge('feature_max', 'Maximum value of input features', ['feature'])
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift detection score', ['feature'])
DATA_DRIFT_ALERT = Gauge('data_drift_alert', 'Data drift alert (1=drift detected)')

# ============================================================
# Configuration
# ============================================================

# Model Registry name (must match train.py)
MODEL_NAME = "NanofluidHeatTransfer"

# Performance monitoring settings
PERFORMANCE_THRESHOLD = 0.95  # R2 threshold for auto-retrain alert
AUTO_RETRAIN_THRESHOLD = 0.90  # R2 threshold for automatic retraining

# Data drift settings
DRIFT_THRESHOLD = 2.0  # Standard deviations from baseline
DRIFT_WINDOW_SIZE = 100  # Number of samples for drift detection

# ============================================================
# Global State
# ============================================================

# Global model variable
model = None
model_info = {}
is_retraining = False  # Lock to prevent concurrent retraining

# Performance tracking
prediction_errors = []
MAX_ERROR_HISTORY = 100

# Data drift tracking - baseline statistics (will be set on first predictions)
baseline_stats = {
    'eta': {'mean': None, 'std': None, 'count': 0},
    'f': {'mean': None, 'std': None, 'count': 0},
    'f_prime': {'mean': None, 'std': None, 'count': 0}
}
recent_data = {
    'eta': deque(maxlen=DRIFT_WINDOW_SIZE),
    'f': deque(maxlen=DRIFT_WINDOW_SIZE),
    'f_prime': deque(maxlen=DRIFT_WINDOW_SIZE)
}
drift_detected = False


def load_model_from_registry(stage="Production"):
    """
    Load model from MLflow Model Registry by stage.
    
    Args:
        stage: "Production", "Staging", or "None" (latest version)
    
    Returns:
        Loaded model or None if not found
    """
    global model, model_info
    
    try:
        client = MlflowClient()
        
        # Try to load from Model Registry
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        print(f"\n{'='*50}")
        print(f"LOADING FROM MODEL REGISTRY")
        print(f"{'='*50}")
        print(f"  Model Name:  {MODEL_NAME}")
        print(f"  Stage:       {stage}")
        print(f"  URI:         {model_uri}")
        
        # Get model version info
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        target_version = None
        for v in versions:
            if v.current_stage == stage:
                target_version = v
                break
        
        if target_version:
            # Get run info for metrics
            run = client.get_run(target_version.run_id)
            model_type = run.data.params.get("model_type", "Unknown")
            r2_score = run.data.metrics.get("r2", "N/A")
            mse_score = run.data.metrics.get("mse", "N/A")
            
            print(f"  Run ID:      {target_version.run_id}")
            print(f"  Model Type:  {model_type}")
            print(f"  R2 Score:    {r2_score}")
            print(f"  MSE:         {mse_score}")
            print(f"{'='*50}\n")
            
            model_info = {
                "name": MODEL_NAME,
                "run_id": target_version.run_id,
                "model_type": model_type,
                "r2": r2_score,
                "mse": mse_score
            }
        
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully from Registry!")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not load from Registry ({stage}): {e}")
        return False


def load_model_fallback():
    """Fallback: Load best model by R2 score from experiment runs."""
    global model, model_info
    
    try:
        print("\n📦 Fallback: Loading best model from experiment runs...")
        
        experiment = mlflow.get_experiment_by_name("nanofluid_heat_transfer")
        if experiment is None:
            print("Experiment not found.")
            return False

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["metrics.r2 DESC"],
            max_results=1
        )
        if runs.empty:
            print("No runs found.")
            return False
            
        best_run = runs.iloc[0]
        best_run_id = best_run.run_id
        experiment_id = experiment.experiment_id
        
        model_type = best_run.get("params.model_type", "Unknown")
        r2_score = best_run.get("metrics.r2", "N/A")
        mse_score = best_run.get("metrics.mse", "N/A")
        
        print(f"  Model Type: {model_type}")
        print(f"  R2 Score:   {r2_score}")
        print(f"  MSE:        {mse_score}")
        
        mlruns_path = os.environ.get("MLRUNS_PATH", "./mlruns")
        artifact_path = None
        
        # Try new MLflow structure
        outputs_path = f"{mlruns_path}/{experiment_id}/{best_run_id}/outputs"
        if os.path.exists(outputs_path):
            model_outputs = glob.glob(f"{outputs_path}/m-*")
            if model_outputs:
                model_id = os.path.basename(model_outputs[0])
                candidate_path = f"{mlruns_path}/{experiment_id}/models/{model_id}/artifacts"
                if os.path.exists(candidate_path):
                    artifact_path = candidate_path
        
        # Fallback to old structure
        if artifact_path is None:
            old_path = f"{mlruns_path}/{experiment_id}/{best_run_id}/artifacts/model"
            if os.path.exists(old_path):
                artifact_path = old_path
        
        if artifact_path is None:
            print(f"Model not found for run {best_run_id}")
            return False
        
        model = mlflow.sklearn.load_model(artifact_path)
        model_info = {
            "name": MODEL_NAME,
            "run_id": best_run_id,
            "model_type": model_type,
            "r2": r2_score,
            "mse": mse_score
        }
        print("✅ Model loaded successfully (fallback)!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback loading failed: {e}")
        return False


def load_model():
    """
    Load model with priority:
    1. Production stage from Model Registry
    2. Staging stage from Model Registry
    3. Fallback: Best R2 from experiment runs
    """
    # Try Production first
    if load_model_from_registry("Production"):
        return
    
    # Try Staging
    if load_model_from_registry("Staging"):
        return
    
    # Fallback to experiment runs
    load_model_fallback()

@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def record_metrics(response):
    resp_time = time.time() - request.start_time
    REQUEST_LATENCY.labels(endpoint=request.path).observe(resp_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path, http_status=response.status_code).inc()
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global drift_detected
    
    if not model:
        ERROR_COUNT.labels(error_type='model_not_loaded').inc()
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Expected features: eta, f, f_prime
        required_cols = ['eta', 'f', 'f_prime']
        if not all(col in df.columns for col in required_cols):
            ERROR_COUNT.labels(error_type='missing_columns').inc()
            return jsonify({'error': f'Missing columns. Required: {required_cols}'}), 400
        
        # Track input data for drift detection
        for col in required_cols:
            for val in df[col].values:
                recent_data[col].append(float(val))
        
        # Update feature statistics
        update_feature_statistics(df[required_cols])
        
        # Check for data drift
        drift_scores = check_data_drift()
        
        # Make prediction with timing
        start_time = time.time()
        predictions = model.predict(df[required_cols])
        prediction_time = time.time() - start_time
        
        # Record metrics
        PREDICTION_COUNT.inc(len(predictions))
        PREDICTION_LATENCY.observe(prediction_time)
        
        for pred in predictions:
            PREDICTION_VALUE.observe(float(pred))
        
        response = {'predictions': predictions.tolist()}
        
        # Add drift warning if detected
        if drift_detected:
            response['drift_warning'] = True
            response['drift_scores'] = drift_scores
        
        return jsonify(response)
        
    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        return jsonify({'error': str(e)}), 500


def update_feature_statistics(df):
    """Update running statistics for features."""
    for col in df.columns:
        values = df[col].values
        
        # Update baseline if not set
        if baseline_stats[col]['mean'] is None:
            baseline_stats[col]['mean'] = float(np.mean(values))
            baseline_stats[col]['std'] = float(np.std(values)) if len(values) > 1 else 1.0
            baseline_stats[col]['count'] = len(values)
        else:
            # Update with exponential moving average
            alpha = 0.1
            baseline_stats[col]['mean'] = (1 - alpha) * baseline_stats[col]['mean'] + alpha * float(np.mean(values))
            if len(values) > 1:
                baseline_stats[col]['std'] = (1 - alpha) * baseline_stats[col]['std'] + alpha * float(np.std(values))
            baseline_stats[col]['count'] += len(values)
        
        # Update Prometheus metrics
        current_mean = float(np.mean(values))
        current_std = float(np.std(values)) if len(values) > 1 else 0
        FEATURE_MEAN.labels(feature=col).set(current_mean)
        FEATURE_STD.labels(feature=col).set(current_std)
        FEATURE_MIN.labels(feature=col).set(float(np.min(values)))
        FEATURE_MAX.labels(feature=col).set(float(np.max(values)))


def check_data_drift():
    """Check for data drift using statistical tests."""
    global drift_detected
    drift_scores = {}
    any_drift = False
    
    for col in ['eta', 'f', 'f_prime']:
        if len(recent_data[col]) < 10 or baseline_stats[col]['mean'] is None:
            drift_scores[col] = 0.0
            continue
        
        # Calculate z-score of recent mean compared to baseline
        recent_mean = np.mean(list(recent_data[col]))
        baseline_mean = baseline_stats[col]['mean']
        baseline_std = baseline_stats[col]['std']
        
        if baseline_std > 0:
            z_score = abs(recent_mean - baseline_mean) / baseline_std
        else:
            z_score = 0.0
        
        drift_scores[col] = float(z_score)
        DATA_DRIFT_SCORE.labels(feature=col).set(z_score)
        
        if z_score > DRIFT_THRESHOLD:
            any_drift = True
    
    drift_detected = any_drift
    DATA_DRIFT_ALERT.set(1 if any_drift else 0)
    
    return drift_scores

@app.route('/model-info')
def get_model_info():
    """Return information about the currently loaded model."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Add performance status
    info = model_info.copy()
    r2 = float(info.get('r2', 0))
    
    if r2 >= PERFORMANCE_THRESHOLD:
        info['status'] = 'good'
    elif r2 >= AUTO_RETRAIN_THRESHOLD:
        info['status'] = 'warning'
    else:
        info['status'] = 'poor'
    
    info['needs_retrain'] = r2 < PERFORMANCE_THRESHOLD
    
    return jsonify(info)


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining.
    This will train all models and promote the best one to Staging.
    """
    global is_retraining
    
    if is_retraining:
        return jsonify({'error': 'Retraining already in progress'}), 409
    
    try:
        is_retraining = True
        RETRAIN_IN_PROGRESS.set(1)
        retrain_start = time.time()
        
        print("\n" + "="*50)
        print("RETRAINING TRIGGERED FROM WEB UI")
        print("="*50)
        
        # Import train module
        from train import train as run_training, promote_staging_to_production
        
        # Run training in a way that captures results
        import io
        import sys
        
        # Capture training output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Run training (this will train all models and promote best to Staging)
            run_training(auto_promote_to_production=True)
        finally:
            sys.stdout = old_stdout
        
        training_output = captured_output.getvalue()
        print(training_output)
        
        # Parse results from output
        best_model = "Unknown"
        r2_score = 0.0
        
        for line in training_output.split('\n'):
            if "Best model:" in line:
                # Extract model name and R2
                parts = line.split("Best model:")
                if len(parts) > 1:
                    model_part = parts[1].strip()
                    if "(" in model_part:
                        best_model = model_part.split("(")[0].strip()
                        r2_part = model_part.split("R2:")[1] if "R2:" in model_part else ""
                        r2_score = float(r2_part.replace(")", "").strip()) if r2_part else 0.0
        
        # Record metrics
        retrain_duration = time.time() - retrain_start
        RETRAIN_COUNT.labels(trigger_type='manual').inc()
        RETRAIN_DURATION.observe(retrain_duration)
        
        # Reload the model
        load_model()
        
        print("✅ Retraining complete, model reloaded!")
        
        return jsonify({
            'success': True,
            'message': 'Retraining complete',
            'best_model': best_model,
            'r2': r2_score,
            'duration_seconds': retrain_duration
        })
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        ERROR_COUNT.labels(error_type='retrain_failed').inc()
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        is_retraining = False
        RETRAIN_IN_PROGRESS.set(0)


@app.route('/retrain-status')
def retrain_status():
    """Check if retraining is currently in progress."""
    return jsonify({
        'is_retraining': is_retraining
    })


@app.route('/performance-check', methods=['POST'])
def performance_check():
    """
    Check model performance against actual values.
    If performance drops below threshold, trigger alert or auto-retrain.
    
    Request body: {
        "predictions": [...],
        "actuals": [...],
        "auto_retrain": true/false
    }
    """
    global prediction_errors
    
    try:
        data = request.get_json()
        predictions = np.array(data.get('predictions', []))
        actuals = np.array(data.get('actuals', []))
        auto_retrain = data.get('auto_retrain', False)
        
        if len(predictions) == 0 or len(actuals) == 0:
            return jsonify({'error': 'Missing predictions or actuals'}), 400
        
        if len(predictions) != len(actuals):
            return jsonify({'error': 'Predictions and actuals must have same length'}), 400
        
        # Calculate error metrics
        mse = float(np.mean((predictions - actuals) ** 2))
        mae = float(np.mean(np.abs(predictions - actuals)))
        
        # Calculate R2 for this batch
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Track errors
        prediction_errors.append({
            'timestamp': time.time(),
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
        
        # Keep only recent errors
        if len(prediction_errors) > MAX_ERROR_HISTORY:
            prediction_errors = prediction_errors[-MAX_ERROR_HISTORY:]
        
        # Update Prometheus metrics
        PREDICTION_ERROR.set(mae)
        
        # Check if performance dropped
        needs_retrain = r2 < AUTO_RETRAIN_THRESHOLD
        performance_warning = r2 < PERFORMANCE_THRESHOLD
        
        result = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'needs_retrain': needs_retrain,
            'performance_warning': performance_warning,
            'threshold': PERFORMANCE_THRESHOLD,
            'auto_retrain_threshold': AUTO_RETRAIN_THRESHOLD
        }
        
        # Auto-retrain if requested and needed
        if auto_retrain and needs_retrain and not is_retraining:
            print(f"⚠️ Performance dropped (R2: {r2:.4f}), triggering auto-retrain...")
            # Trigger retrain in background thread
            thread = threading.Thread(target=trigger_background_retrain)
            thread.start()
            result['retrain_triggered'] = True
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def trigger_background_retrain():
    """Trigger retraining in background."""
    global is_retraining
    
    if is_retraining:
        return
    
    try:
        is_retraining = True
        RETRAIN_IN_PROGRESS.set(1)
        retrain_start = time.time()
        
        print("\n" + "="*50)
        print("AUTO-RETRAIN TRIGGERED (Performance Drop)")
        print("="*50)
        
        from train import train as run_training
        run_training(auto_promote_to_production=True)
        
        # Record metrics
        RETRAIN_COUNT.labels(trigger_type='auto').inc()
        RETRAIN_DURATION.observe(time.time() - retrain_start)
        
        # Reload model
        load_model()
        print("✅ Auto-retrain complete!")
        
    except Exception as e:
        print(f"❌ Auto-retrain failed: {e}")
        ERROR_COUNT.labels(error_type='auto_retrain_failed').inc()
    finally:
        is_retraining = False
        RETRAIN_IN_PROGRESS.set(0)


@app.route('/monitoring')
def monitoring_dashboard():
    """Return comprehensive monitoring data as JSON."""
    # Calculate drift scores
    drift_scores = {}
    for col in ['eta', 'f', 'f_prime']:
        if len(recent_data[col]) >= 10 and baseline_stats[col]['mean'] is not None:
            recent_mean = np.mean(list(recent_data[col]))
            baseline_mean = baseline_stats[col]['mean']
            baseline_std = baseline_stats[col]['std']
            if baseline_std > 0:
                drift_scores[col] = abs(recent_mean - baseline_mean) / baseline_std
            else:
                drift_scores[col] = 0.0
        else:
            drift_scores[col] = 0.0
    
    # Get recent errors
    recent_errors = prediction_errors[-10:] if prediction_errors else []
    
    return jsonify({
        'model': {
            'name': model_info.get('name', 'Unknown'),
            'run_id': model_info.get('run_id', 'Unknown'),
            'algorithm': model_info.get('model_type', 'Unknown'),
            'r2_score': model_info.get('r2', 0),
            'mse_score': model_info.get('mse', 0),
            'loaded': model is not None
        },
        'performance': {
            'threshold': PERFORMANCE_THRESHOLD,
            'auto_retrain_threshold': AUTO_RETRAIN_THRESHOLD,
            'current_r2': model_info.get('r2', 0),
            'status': 'good' if float(model_info.get('r2', 0)) >= PERFORMANCE_THRESHOLD else 
                     ('warning' if float(model_info.get('r2', 0)) >= AUTO_RETRAIN_THRESHOLD else 'poor')
        },
        'data_drift': {
            'detected': drift_detected,
            'threshold': DRIFT_THRESHOLD,
            'scores': drift_scores,
            'baseline_stats': baseline_stats
        },
        'retraining': {
            'in_progress': is_retraining
        },
        'recent_errors': recent_errors,
        'feature_statistics': {
            col: {
                'recent_count': len(recent_data[col]),
                'baseline_mean': baseline_stats[col]['mean'],
                'baseline_std': baseline_stats[col]['std'],
                'total_samples': baseline_stats[col]['count']
            }
            for col in ['eta', 'f', 'f_prime']
        }
    })


@app.route('/metrics')
def metrics():
    # Update model metrics
    if model_info:
        try:
            if 'r2' in model_info:
                MODEL_R2_SCORE.set(float(model_info['r2']))
            if 'mse' in model_info:
                MODEL_MSE_SCORE.set(float(model_info['mse']))
            
            MODEL_INFO.info({
                'name': str(model_info.get('name', 'unknown')),
                'run_id': str(model_info.get('run_id', 'unknown')),
                'algorithm': str(model_info.get('model_type', 'unknown'))
            })
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    # Return with correct Content-Type for Prometheus
    from flask import Response
    return Response(generate_latest(), mimetype='text/plain; version=0.0.4; charset=utf-8')

if __name__ == '__main__':
    load_model()
    # In production with gunicorn, load_model() should be called differently or on module load if possible.
    # But for this simple setup:
    app.run(host='0.0.0.0', port=5000)
