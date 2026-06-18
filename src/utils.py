import os
import joblib

def save_object(file_path, obj):
    """Saves a Python object (like a model or scaler) to a physical file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        print(f"Error saving object: {e}")
        raise e

def load_object(file_path):
    """Loads a physical file back into a Python object."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading object: {e}")
        raise e