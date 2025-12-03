import os
import sys
import pandas as pd
import joblib
import boto3
import tarfile
import shutil
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import sagemaker
from sagemaker import Model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================
# CONFIG VARIABLES
# ================
S3_BUCKET = "rain-prediction-autodeploy-mumbai"
ENDPOINT_NAME = "rain-prediction-autodeploy-mumbai"
REGION = "ap-south-1"
SAGEMAKER_ROLE = "arn:aws:iam::493272324412:role/service-role/AmazonSageMaker-ExecutionRole-20251203T141656"

# S3 paths
MODEL_TAR_S3_KEY = "model/model.tar.gz"

# Local paths
TEMP_DIR = "/tmp/partial_fit_temp"
MODEL_TAR_LOCAL = os.path.join(TEMP_DIR, "model.tar.gz")
EXTRACT_DIR = os.path.join(TEMP_DIR, "extracted")
UPDATED_MODEL_DIR = os.path.join(TEMP_DIR, "updated_model")
UPDATED_TAR_PATH = os.path.join(TEMP_DIR, "updated_model.tar.gz")

# CSV folder in repo
INPUT_DATA_PATH = "csv_folder"

# Ensure temp directories exist
for dir_path in [TEMP_DIR, EXTRACT_DIR, UPDATED_MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def clean_temp_directories():
    """Clean temporary directories"""
    for dir_path in [TEMP_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def download_and_extract_model(s3_client):
    """Download model.tar.gz from S3 and extract it"""
    logger.info(f"Downloading model from s3://{S3_BUCKET}/{MODEL_TAR_S3_KEY}")
    
    try:
        # Download model.tar.gz
        s3_client.download_file(S3_BUCKET, MODEL_TAR_S3_KEY, MODEL_TAR_LOCAL)
        logger.info(f"✓ Downloaded model.tar.gz ({os.path.getsize(MODEL_TAR_LOCAL) / 1024:.2f} KB)")
        
        # Extract tar.gz
        with tarfile.open(MODEL_TAR_LOCAL, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        
        # List extracted files
        extracted_files = os.listdir(EXTRACT_DIR)
        logger.info(f"Extracted files: {extracted_files}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download/extract model: {str(e)}")
        return False


def load_model_and_scaler():
    """Load model and scaler from extracted files"""
    model_path = os.path.join(EXTRACT_DIR, "sgd_model.joblib")
    scaler_path = os.path.join(EXTRACT_DIR, "scaler.joblib")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return None, None
    
    try:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("✓ Loaded existing model and scaler")
        return clf, scaler
    except Exception as e:
        logger.error(f"Failed to load model/scaler: {str(e)}")
        return None, None


def load_csv_data():
    """Load the latest CSV file from csv_folder"""
    csv_files = [f for f in os.listdir(INPUT_DATA_PATH) if f.endswith(".csv")]
    
    if not csv_files:
        logger.error(f"❌ No CSV files found in {INPUT_DATA_PATH}/")
        return None, None
    
    # Use the latest CSV file
    csv_files.sort(reverse=True)
    csv_path = os.path.join(INPUT_DATA_PATH, csv_files[0])
    logger.info(f"Using CSV file: {csv_path}")
    
    try:
        data = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ["Date", "Location", "Rain Tomorrow"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None, None
        
        # Prepare features and target
        X = data.drop(columns=["Date", "Location", "Rain Tomorrow"])
        y = data["Rain Tomorrow"]
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Failed to load CSV data: {str(e)}")
        return None, None


def create_updated_tar_gz(clf, scaler):
    """Create updated model.tar.gz with new model and scaler"""
    try:
        # Save updated model and scaler
        joblib.dump(clf, os.path.join(UPDATED_MODEL_DIR, "sgd_model.joblib"))
        joblib.dump(scaler, os.path.join(UPDATED_MODEL_DIR, "scaler.joblib"))
        
        # Copy inference.py from extracted files or create new
        inference_src = os.path.join(EXTRACT_DIR, "inference.py")
        inference_dst = os.path.join(UPDATED_MODEL_DIR, "inference.py")
        
        if os.path.exists(inference_src):
            shutil.copy2(inference_src, inference_dst)
            logger.info("✓ Copied inference.py from existing model")
        else:
            logger.warning("inference.py not found in extracted files")
            # Create basic inference.py
            create_basic_inference_file(inference_dst)
        
        # Create new tar.gz
        with tarfile.open(UPDATED_TAR_PATH, "w:gz") as tar:
            tar.add(os.path.join(UPDATED_MODEL_DIR, "sgd_model.joblib"), arcname="sgd_model.joblib")
            tar.add(os.path.join(UPDATED_MODEL_DIR, "scaler.joblib"), arcname="scaler.joblib")
            tar.add(os.path.join(UPDATED_MODEL_DIR, "inference.py"), arcname="inference.py")
        
        logger.info(f"✓ Created updated model.tar.gz ({os.path.getsize(UPDATED_TAR_PATH) / 1024:.2f} KB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create updated tar.gz: {str(e)}")
        return False


def create_basic_inference_file(file_path):
    """Create a basic inference.py file"""
    inference_code = '''import joblib
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
        instances = data.get("instances", [])
        return np.array(instances)
    elif request_content_type == 'text/csv':
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
    
    scaled_data = scaler.transform(input_data)
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
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
    
    with open(file_path, "w") as f:
        f.write(inference_code)
    logger.info(f"Created basic inference.py at {file_path}")


def upload_to_s3(s3_client):
    """Upload updated model.tar.gz to S3"""
    try:
        s3_client.upload_file(UPDATED_TAR_PATH, S3_BUCKET, MODEL_TAR_S3_KEY)
        logger.info(f"✓ Uploaded updated model to s3://{S3_BUCKET}/{MODEL_TAR_S3_KEY}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        return False


def update_sagemaker_endpoint():
    """Update SageMaker endpoint with new model"""
    try:
        logger.info(f"Updating SageMaker endpoint: {ENDPOINT_NAME}")
        
        session = sagemaker.Session(
            boto_session=boto3.Session(region_name=REGION)
        )
        
        # Create model object
        sm_model = Model(
            model_data=f"s3://{S3_BUCKET}/{MODEL_TAR_S3_KEY}",
            role=SAGEMAKER_ROLE,
            sagemaker_session=session,
            entry_point="inference.py"
        )
        
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
        
        # Deploy/update endpoint
        sm_model.deploy(
            initial_instance_count=1,
            instance_type="ml.t2.medium",
            endpoint_name=ENDPOINT_NAME,
            update_endpoint=update_endpoint,
            wait=False  # Don't wait (deployment takes time)
        )
        
        logger.info(f"✓ Endpoint '{ENDPOINT_NAME}' update initiated")
        logger.info("Note: Endpoint deployment may take 5-10 minutes to complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update SageMaker endpoint: {str(e)}")
        return False


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STARTING PARTIAL FIT AUTOMATION")
    logger.info("=" * 60)
    
    # Clean temp directories
    clean_temp_directories()
    
    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=REGION)
    
    try:
        # Step 1: Download and extract model from S3
        if not download_and_extract_model(s3_client):
            logger.error("Failed to download/extract model. Exiting.")
            sys.exit(1)
        
        # Step 2: Load model and scaler
        clf, scaler = load_model_and_scaler()
        if clf is None or scaler is None:
            logger.error("Failed to load model/scaler. Exiting.")
            sys.exit(1)
        
        # Step 3: Load CSV data
        X, y = load_csv_data()
        if X is None or y is None:
            logger.error("Failed to load CSV data. Exiting.")
            sys.exit(1)
        
        # Step 4: Partial fit with new data
        logger.info("Performing partial fit with new data...")
        X_scaled = scaler.transform(X)
        clf.partial_fit(X_scaled, y)
        logger.info("✓ Partial fit completed")
        
        # Step 5: Create updated model.tar.gz
        if not create_updated_tar_gz(clf, scaler):
            logger.error("Failed to create updated model.tar.gz. Exiting.")
            sys.exit(1)
        
        # Step 6: Upload to S3
        if not upload_to_s3(s3_client):
            logger.error("Failed to upload to S3. Exiting.")
            sys.exit(1)
        
        # Step 7: Update SageMaker endpoint
        if not update_sagemaker_endpoint():
            logger.warning("Endpoint update failed, but model is updated in S3")
        
        logger.info("=" * 60)
        logger.info("✅ PARTIAL FIT AUTOMATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model updated: s3://{S3_BUCKET}/{MODEL_TAR_S3_KEY}")
        logger.info(f"Endpoint update initiated: {ENDPOINT_NAME}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
