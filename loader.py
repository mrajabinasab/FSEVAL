import os
import scipy.io
import numpy as np
import pandas as pd

def load_dataset(dataset_name, data_dir="datasets"):
    """
    Loads a dataset from either a .mat file or two .csv files (X and y).
    
    CSV Format expected:
        - {dataset_name}_X.csv
        - {dataset_name}_y.csv
    """
    
    # 1. Try Loading from .mat file first
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if os.path.exists(mat_path):
        try:
            print(f"Loading {dataset_name}.mat...")
            mat_data = scipy.io.loadmat(mat_path)
            
            if 'X' in mat_data and 'Y' in mat_data:
                X = mat_data['X']
                y = mat_data['Y'].ravel()
                return _validate_and_return(X, y, dataset_name)
            else:
                print(f"SKIPPING {dataset_name}: .mat file missing 'X' or 'Y' keys.")
        except Exception as e:
            print(f"FAILED to process .mat for {dataset_name}: {e}")

    # 2. Try Loading from .csv files if .mat doesn't exist or failed
    csv_x_path = os.path.join(data_dir, f"{dataset_name}_X.csv")
    csv_y_path = os.path.join(data_dir, f"{dataset_name}_y.csv")
    
    if os.path.exists(csv_x_path) and os.path.exists(csv_y_path):
        try:
            print(f"Loading CSV files for {dataset_name}...")
            # Using pandas for robust CSV reading, then converting to numpy
            X = pd.read_csv(csv_x_path).to_numpy()
            y = pd.read_csv(csv_y_path).to_numpy().ravel()
            return _validate_and_return(X, y, dataset_name)
        except Exception as e:
            print(f"FAILED to process CSV for {dataset_name}: {e}")

    print(f"Error: No valid .mat or .csv pairs found for '{dataset_name}' in {data_dir}")
    return None, None

def _validate_and_return(X, y, name):
    """Internal helper to validate dimensions and shapes."""
    if X.ndim != 2:
        print(f"SKIPPING {name}: 'X' is not 2D (shape: {X.shape}).")
        return None, None
        
    if X.shape[0] != y.shape[0]:
        print(f"SKIPPING {name}: Sample mismatch. X: {X.shape[0]}, y: {y.shape[0]}")
        return None, None

    print(f"  Successfully loaded '{name}' | X: {X.shape} | y: {y.shape}")
    return X, y