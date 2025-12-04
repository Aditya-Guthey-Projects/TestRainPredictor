import os
import sys
import subprocess
import json
import traceback
import numpy as np

# Debug: Print Python and initial sklearn version
print(f"Python version: {sys.version}")
try:
    import sklearn
    print(f"Initial sklearn version: {sklearn.__version__}")
except ImportError as e:
    print(f"sklearn not found initially: {e}")

# Try to ensure we have sklearn 1.7.2
def ensure_correct_versions():
    """Ensure we have the correct package versions."""
    try:
        import sklearn
        if sklearn.__version__ != '1.7.2':
            print(f"WARNING: sklearn version {sklearn.__version__} detected, installing 1.7.2...")
            # Install correct version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "scikit-learn==1.7.2", "joblib==1.3.2", "--force-reinstall"
            ])
            # Reload module
            import importlib
            import sklearn
            importlib.reload(sklearn)
            print(f"Updated sklearn version: {sklearn.__version__}")
    except Exception as e:
        print(f"Error ensuring sklearn version: {e}")
        traceback.print_exc()

# Call this at module import time
ensure_correct_versions()

# Now import joblib after ensuring sklearn version
import joblib
import sklearn  # Re-import to get the updated version

print(f"Final sklearn version: {sklearn.__version__}")
print(f"Joblib version: {joblib.__version__}")

def model_fn(model_dir):
    """Load the model from the model_dir."""
    print(f"Loading model from: {model_dir}")
    print(f"Files in model_dir: {os.listdir(model_dir)}")
    
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    # Final check of sklearn version before loading
    import sklearn
    print(f"sklearn version at model loading: {sklearn.__version__}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Type: {type(model)}")
        print(f"Model classes: {getattr(model, 'classes_', 'No classes attribute')}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

def input_fn(request_body, request_content_type):
    """Parse input data."""
    print(f"Received content type: {request_content_type}")
    print(f"Request body length: {len(request_body)}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print(f"Parsed JSON data: {data}")
        
        # Handle different input formats
        if "instances" in data:
            instances = data["instances"]
        elif isinstance(data, list):
            instances = data
        else:
            # Assume it's already in the right format
            instances = data
            
        result = np.array(instances)
        print(f"Input shape: {result.shape}")
        return result
        
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(data, model):
    """Make predictions."""
    print(f"Predicting on data with shape: {data.shape}")
    
    preds = model.predict(data)
    print(f"Predictions shape: {preds.shape}")
    
    # Check if model has predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data)
        print(f"Probabilities shape: {probs.shape}")
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }
    else:
        print("Model does not have predict_proba method")
        return {
            "predictions": preds.tolist(),
            "probabilities": []
        }

def output_fn(prediction, content_type):
    """Format predictions for response."""
    print(f"Output content type: {content_type}")
    return json.dumps(prediction)
