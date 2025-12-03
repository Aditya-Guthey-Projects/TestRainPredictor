import os
import sys
import pandas as pd
import joblib
import boto3
import tarfile
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import sagemaker
from sagemaker.model import Model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================
# CONFIG VARIABLES
# ================
S3_BUCKET = os.environ.get("S3_BUCKET", "rain-prediction-autodeploy-mumbai")             # e.g. 'my-bucket'
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "rain-prediction-autodeploy-mumbai")     # e.g. 'weather-endpoint'
REGION = os.environ.get("AWS_REGION", "ap-south-1")  
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::493272324412:role/service-role/AmazonSageMaker-ExecutionRole-20251203T141656")   

# Check for required environment variables
if not SAGEMAKER_ROLE:
    logger.error("SAGEMAKER_ROLE environment variable is not set!")
    sys.exit(1)

# folder inside your GitHub repo containing CSV files
input_data_path = "csv_folder"

# temporary directory inside CodeBuild
model_dir = "/tmp/model"
os.makedirs(model_dir, exist_ok=True)


def main():
    try:
        # ============================
        # 1. LOAD NEW CSV FROM REPO
        # ============================
        csv_files = [f for f in os.listdir(input_data_path) if f.endswith(".csv")]
        
        if not csv_files:
            logger.error("❌ No CSV file found in csv_folder/")
            sys.exit(1)
        
        # Use the latest CSV
        csv_path = os.path.join(input_data_path, sorted(csv_files)[-1])
        logger.info(f"Using CSV: {csv_path}")
        
        data = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ["Date", "Location", "Rain Tomorrow"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)
        
        X = data.drop(columns=["Date", "Location", "Rain Tomorrow"])
        y = data["Rain Tomorrow"]
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # ============================
        # 2. DOWNLOAD EXISTING MODEL FROM S3
        # ============================
        s3 = boto3.client("s3", region_name=REGION)
        
        model_s3_key = "model/sgd_model.joblib"
        scaler_s3_key = "model/scaler.joblib"
        
        model_path = os.path.join(model_dir, "sgd_model.joblib")
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        
        model_exists = False
        try:
            # Check if model exists in S3
            s3.head_object(Bucket=S3_BUCKET, Key=model_s3_key)
            s3.head_object(Bucket=S3_BUCKET, Key=scaler_s3_key)
            
            s3.download_file(S3_BUCKET, model_s3_key, model_path)
            s3.download_file(S3_BUCKET, scaler_s3_key, scaler_path)
            clf = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            model_exists = True
            logger.info("Loaded existing model & scaler from S3")
        except Exception as e:
            logger.warning(f"No existing model found or error loading: {str(e)}")
            logger.info("Creating new model...")
            clf = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
            scaler = StandardScaler()
        
        # ============================
        # 3. TRAIN OR UPDATE MODEL
        # ============================
        if not model_exists:
            # Fit new model
            X_scaled = scaler.fit_transform(X)
            clf.partial_fit(X_scaled, y, classes=[0, 1])
            
            joblib.dump(clf, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Upload to S3
            s3.upload_file(model_path, S3_BUCKET, model_s3_key)
            s3.upload_file(scaler_path, S3_BUCKET, scaler_s3_key)
            
            logger.info("Initial model created and uploaded to S3")
        else:
            # Update existing model with partial fit
            X_scaled = scaler.transform(X)
            clf.partial_fit(X_scaled, y)
            
            # Save local updates
            joblib.dump(clf, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Upload updated model files
            s3.upload_file(model_path, S3_BUCKET, model_s3_key)
            s3.upload_file(scaler_path, S3_BUCKET, scaler_s3_key)
            
            logger.info("Updated model & scaler uploaded to S3")
        
        # ============================
        # 4. CHECK IF INFERENCE.PY EXISTS
        # ============================
        inference_file = "inference.py"
        if not os.path.exists(inference_file):
            logger.error(f"❌ {inference_file} not found in current directory!")
            logger.info("Creating a basic inference.py file...")
            create_basic_inference_file()
        
        # ============================
        # 5. PACKAGE MODEL INTO model.tar.gz
        # ============================
        tar_path = "/tmp/model.tar.gz"
        
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname="sgd_model.joblib")
            tar.add(scaler_path, arcname="scaler.joblib")
            tar.add(inference_file, arcname="inference.py")
        
        model_tar_s3_key = "model/model.tar.gz"
        s3.upload_file(tar_path, S3_BUCKET, model_tar_s3_key)
        
        logger.info("model.tar.gz created & uploaded to S3")
        
        # ============================
        # 6. UPDATE EXISTING ENDPOINT
        # ============================
        logger.info(f"Setting up SageMaker session in region: {REGION}")
        session = sagemaker.Session(
            boto_session=boto3.Session(region_name=REGION)
        )
        
        sm_model = Model(
            model_data=f"s3://{S3_BUCKET}/{model_tar_s3_key}",
            role=SAGEMAKER_ROLE,
            sagemaker_session=session,
            entry_point="inference.py"
        )
        
        logger.info(f"Updating endpoint: {ENDPOINT_NAME} ...")
        
        # Check if endpoint exists
        sm_client = boto3.client("sagemaker", region_name=REGION)
        try:
            sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
            update_endpoint = True
            logger.info(f"Endpoint '{ENDPOINT_NAME}' exists, updating...")
        except sm_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                update_endpoint = False
                logger.info(f"Endpoint '{ENDPOINT_NAME}' doesn't exist, creating new...")
            else:
                raise
        
        sm_model.deploy(
            initial_instance_count=1,
            instance_type="ml.t2.medium",
            endpoint_name=ENDPOINT_NAME,
            update_endpoint=update_endpoint,
            wait=False  # Don't wait for deployment to complete
        )
        
        logger.info("\n✅ Partial fit complete!")
        logger.info(f"➡️ Endpoint '{ENDPOINT_NAME}' is being {'updated' if update_endpoint else 'created'}.")
        
    except Exception as e:
        logger.error(f"❌ Error in partial_fit_automate.py: {str(e)}", exc_info=True)
        sys.exit(1)


def create_basic_inference_file():
    """Create a basic inference.py file if it doesn't exist"""
    inference_code = '''import joblib
import numpy as np
import os

def model_fn(model_dir):
    """Load model and scaler from model_dir"""
    clf = joblib.load(os.path.join(model_dir, "sgd_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return {"model": clf, "scaler": scaler}

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        import json
        data = json.loads(request_body)
        # Expecting format: {"instances": [[feature1, feature2, ...]]}
        return np.array(data.get("instances", []))
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
        import json
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
    
    with open("inference.py", "w") as f:
        f.write(inference_code)
    logger.info("Created basic inference.py file")


if __name__ == "__main__":
    main()
