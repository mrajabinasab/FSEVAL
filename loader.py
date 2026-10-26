import os
import scipy.io
import pandas as pd


def load_dataset(dataset_name, data_dir="datasets", verbose=True):
    """
    Loads a dataset from either a .mat file or two .csv files (X and y).

    .mat format expected:
        - {dataset_name}.mat with keys 'X' and 'Y'
    CSV format expected (each with a header row):
        - {dataset_name}_X.csv
        - {dataset_name}_y.csv

    Returns (X, y), or (None, None) if the dataset cannot be loaded.
    """
    X, y, messages = _load(dataset_name, data_dir)
    if verbose:
        for msg in messages:
            print(msg)
    return X, y


def _load(dataset_name, data_dir):
    """Load without printing. Returns (X, y, messages); X and y are None on failure."""
    messages = []

    # 1. Try loading from .mat file first
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if os.path.exists(mat_path):
        try:
            mat_data = scipy.io.loadmat(mat_path)
            if "X" in mat_data and "Y" in mat_data:
                X, y, msg = _validate(mat_data["X"], mat_data["Y"].ravel(), dataset_name)
                messages.append(msg)
                if X is not None:
                    return X, y, messages
            else:
                messages.append(f"SKIPPING {dataset_name}: .mat file missing 'X' or 'Y' keys.")
        except Exception as e:
            messages.append(f"FAILED to process .mat for {dataset_name}: {e}")

    # 2. Try loading from .csv files if .mat doesn't exist or failed
    csv_x_path = os.path.join(data_dir, f"{dataset_name}_X.csv")
    csv_y_path = os.path.join(data_dir, f"{dataset_name}_y.csv")
    if os.path.exists(csv_x_path) and os.path.exists(csv_y_path):
        try:
            X = pd.read_csv(csv_x_path).to_numpy()
            y = pd.read_csv(csv_y_path).to_numpy().ravel()
            X, y, msg = _validate(X, y, dataset_name)
            messages.append(msg)
            if X is not None:
                return X, y, messages
        except Exception as e:
            messages.append(f"FAILED to process CSV for {dataset_name}: {e}")

    if not messages:
        messages.append(f"No .mat or .csv pair found for '{dataset_name}' in {data_dir}")
    return None, None, messages


def _validate(X, y, name):
    """Validate dimensions and shapes. Returns (X, y, message); X and y are None if invalid."""
    if X.ndim != 2:
        return None, None, f"SKIPPING {name}: 'X' is not 2D (shape: {X.shape})."
    if X.shape[0] != y.shape[0]:
        return None, None, f"SKIPPING {name}: Sample mismatch. X: {X.shape[0]}, y: {y.shape[0]}"
    return X, y, f"Successfully loaded '{name}' | X: {X.shape} | y: {y.shape}"
