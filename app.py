from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest
import time
import os

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('webapp_request_count', 'Total Web App Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('webapp_request_latency_seconds', 'Request latency', ['endpoint'])

# Global model variable
model = None

def load_model():
    global model
    try:
        # Load latest run from experiment
        experiment = mlflow.get_experiment_by_name("nanofluid_heat_transfer")
        if experiment is None:
            print("Experiment not found.")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            print("No runs found.")
            return
            
        latest_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"Loading model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")

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
    return jsonify({
        "message": "Welcome to Nanofluid Heat Transfer Prediction API",
        "endpoints": {
            "/predict" : "POST - Send JSON with keys: eta, f, f_prime",
            "/metrics" : "GET - Prometheus metrics"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Expected features: eta, f, f_prime
        required_cols = ['eta', 'f', 'f_prime']
        if not all(col in df.columns for col in required_cols):
             return jsonify({'error': f'Missing columns. Required: {required_cols}'}), 400
             
        predictions = model.predict(df[required_cols])
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest()

if __name__ == '__main__':
    load_model()
    # In production with gunicorn, load_model() should be called differently or on module load if possible.
    # But for this simple setup:
    app.run(host='0.0.0.0', port=5000)
