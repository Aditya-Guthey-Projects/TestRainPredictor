import os
import joblib
import json
import numpy as np
import pandas as pd

# -------------------------------
# 1. Load model and scaler
# -------------------------------

def model_fn(model_dir):
    """
    SageMaker loads model.tar.gz into model_dir.
    sgd_model.joblib and scaler.joblib must be inside.
    """
    model_path = os.path.join(model_dir, "sgd_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return {"model": clf, "scaler": scaler}


# -------------------------------
# 2. Input handler
# -------------------------------

def input_fn(request_body, content_type):
    """
    Expected input format:
    JSON:
    {
        "data": {
            "MinTemp": value,
            "MaxTemp": value,
            ...
        }
    }
    """
    if content_type == "application/json":
        body = json.loads(request_body)

        # Parse provided data
        if "data" not in body:
            raise ValueError("JSON must contain a 'data' field")

        data = body["data"]

        # Convert dict → pandas dataframe with one row
        df = pd.DataFrame([data])

        return df

    else:
        raise ValueError(f"Content type {content_type} not supported")


# -------------------------------
# 3. Prediction
# -------------------------------

def predict_fn(input_data, model_objects):
    clf = model_objects["model"]
    scaler = model_objects["scaler"]

    # Scale input features
    X_scaled = scaler.transform(input_data)

    # Prediction
    pred = clf.predict(X_scaled)
    proba = clf.predict_proba(X_scaled)[0][1]  # probability of class 1

    result = {
        "prediction": int(pred[0]),
        "probability_of_rain": float(proba)
    }

    return result


# -------------------------------
# 4. Output handler
# -------------------------------

def output_fn(prediction_output, accept):
    """
    Format final response as JSON
    """
    if accept == "application/json":
        return json.dumps(prediction_output), accept

    raise ValueError(f"Accept {accept} not supported")
