import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

def train():
    # Set MLflow experiment
    mlflow.set_experiment("nanofluid_heat_transfer")
    
    with mlflow.start_run():
        # Load data
        data_path = "data/nanofluid_data.csv"
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Simple feature engineering/selection
        # Target: theta_prime (related to heat transfer)
        # Features: eta, f, f_prime (related to flow)
        # Note: This is a simplified physics approximation for demo/MLOps
        
        X = df[['eta', 'f', 'f_prime']]
        y = df['theta_prime']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Params
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        print("Training model...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print("Model saved to MLflow.")

if __name__ == "__main__":
    train()
