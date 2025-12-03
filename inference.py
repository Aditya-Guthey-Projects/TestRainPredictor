import joblib
import numpy as np
import os
import json
import sys

# Add scikit-learn to path for compatibility
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Warning: Could not import scikit-learn directly")


def model_fn(model_dir):
    """Load model and scaler from model_dir"""
    print(f"Loading model from: {model_dir}")
    print(f"Files in model_dir: {os.listdir(model_dir)}")
    
    try:
        # Load the model using joblib
        model_path = os.path.join(model_dir, "sgd_model.joblib")
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        
        # IMPORTANT: Use joblib.load with custom handling
        import joblib
        
        # First try normal loading
        try:
            clf = joblib.load(model_path)
            print("Model loaded successfully with normal joblib.load")
        except Exception as e:
            print(f"Normal load failed: {e}. Trying with custom unpickler...")
            # Use custom unpickler for compatibility
            from sklearn.utils._joblib import _LazyImport
            import pickle
            
            # Create a custom unpickler that handles scikit-learn modules
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle _loss module issue
                    if module == 'sklearn.linear_model._stochastic_gradient' and name == '_loss':
                        # Return a dummy _loss module
                        class DummyLoss:
                            pass
                        return DummyLoss
                    # Handle other scikit-learn imports
                    if module.startswith('sklearn.'):
                        # Map to correct module
                        module = module.replace('_stochastic_gradient', 'linear_model._stochastic_gradient')
                    return super().find_class(module, name)
            
            with open(model_path, 'rb') as f:
                clf = CustomUnpickler(f).load()
            print("Model loaded with custom unpickler")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully")
        
        # Test prediction to verify model works
        print(f"Model type: {type(clf)}")
        print(f"Model classes: {clf.classes_ if hasattr(clf, 'classes_') else 'No classes'}")
        
        return {"model": clf, "scaler": scaler}
        
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        raise


def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        instances = data.get("instances", [])
        
        # Convert to numpy array
        result = np.array(instances, dtype=np.float32)
        print(f"Parsed input shape: {result.shape}")
        return result
        
    elif request_content_type == 'text/csv':
        import io
        import csv
        
        # Parse CSV data
        f = io.StringIO(request_body)
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append([float(x) for x in row])
        
        result = np.array(data, dtype=np.float32)
        print(f"Parsed CSV input shape: {result.shape}")
        return result
        
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Make predictions"""
    print(f"Making prediction for input shape: {input_data.shape}")
    
    clf = model_dict["model"]
    scaler = model_dict["scaler"]
    
    # Ensure 2D array
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    
    print(f"Reshaped input: {input_data.shape}")
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    print(f"Scaled data shape: {scaled_data.shape}")
    
    # Make predictions
    try:
        predictions = clf.predict(scaled_data)
        print(f"Predictions: {predictions}")
        
        # Get probabilities if available
        if hasattr(clf, 'predict_proba'):
            probabilities = clf.predict_proba(scaled_data)
            print(f"Probabilities shape: {probabilities.shape}")
        else:
            probabilities = []
            print("Model does not support predict_proba")
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if len(probabilities) > 0 else []
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}", file=sys.stderr)
        raise


def output_fn(prediction, accept):
    """Format output"""
    print(f"Formatting output for accept type: {accept}")
    
    if accept == "application/json":
        response = json.dumps(prediction)
        print(f"Response length: {len(response)}")
        return response, accept
        
    elif accept == "text/csv":
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        predictions = prediction["predictions"]
        probabilities = prediction.get("probabilities", [])
        
        if probabilities:
            for pred, probs in zip(predictions, probabilities):
                writer.writerow([pred] + list(probs))
        else:
            for pred in predictions:
                writer.writerow([pred])
        
        result = output.getvalue()
        print(f"CSV response length: {len(result)}")
        return result, accept
        
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
