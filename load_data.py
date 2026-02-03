import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil
import os
import pandas as pd

def load_and_save_data():
    # Load the latest version
    print("Downloading dataset from KaggleHub...")
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("rajamujtabaahmed/nanofluid-heat-transfer-over-a-stretching-sheet")
    
    print("Path to dataset files:", path)
    
    # Find the CSV file in the downloaded path
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")
    
    csv_path = os.path.join(path, csv_files[0])
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Dataset loaded successfully.")
    print("First 5 records:", df.head())
    
    # Define output path
    output_dir = "data"
    output_file = os.path.join(output_dir, "nanofluid_data.csv")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    print(f"Saving data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Data saved.")

if __name__ == "__main__":
    load_and_save_data()
