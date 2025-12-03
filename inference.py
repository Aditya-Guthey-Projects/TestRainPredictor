import joblib
import numpy as np
import os
import json

def model_fn(model_dir):
    """Load model and scaler from model_dir"""
    try:
        clf = joblib.load(os.path.join(model_dir, "sgd_model.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        return {"model": clf, "scaler": scaler}
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Expecting format: {"instances": [[feature1, feature2, ...]]}
        instances = data.get("instances", [])
        return np.array(instances)
    elif request_content_type == 'text/csv':
        # CSV format: feature1,feature2,... per line
        import io
        import csv
        f = io.StringIO(request_body)
        return np.array(list(csv.reader(f)), dtype=float)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions"""
    clf = model_dict["model"]
    scaler = model_dict["scaler"]
    
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Make predictions
    predictions = clf.predict(scaled_data)
    probabilities = clf.predict_proba(scaled_data)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }

def output_fn(prediction, accept):
    """Format output"""
    if accept == "application/json":
        return json.dumps(prediction), accept
    elif accept == "text/csv":
        # Format as CSV: prediction,prob_class0,prob_class1
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        for pred, probs in zip(prediction["predictions"], prediction["probabilities"]):
            writer.writerow([pred] + probs)
        return output.getvalue(), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
