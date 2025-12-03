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

# CSV folder in repo
INPUT_DATA_PATH = "csv_folder"


def clean_and_create_directories():
    """Clean and create all necessary directories"""
    # Base temp directory
    temp_dir = "/tmp/partial_fit_temp"
    
    # Remove if exists and create fresh
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Create directory structure
    directories = [
        temp_dir,
        os.path.join(temp_dir, "extracted"),
        os.path.join(temp_dir, "updated_model")
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    
    return {
        "temp_dir": temp_dir,
        "model_tar_local": os.path.join(temp_dir, "model.tar.gz"),
        "extract_dir": os.path.join(temp_dir, "extracted"),
        "updated_model_dir": os.path.join(temp_dir, "updated_model"),
        "updated_tar_path": os.path.join(temp_dir, "updated_model.tar.gz")
    }


def download_and_extract_model(s3_client, paths):
    """Download model.tar.gz from S3 and extract it"""
    logger.info(f"Downloading model from s3://{S3_BUCKET}/{MODEL_TAR_S3_KEY}")
    
    try:
        # Download model.tar.gz
        s3_client.download_file(S3_BUCKET, MODEL_TAR_S3_KEY, paths["model_tar_local"])
        file_size = os.path.getsize(paths["model_tar_local"])
        logger.info(f"✓ Downloaded model.tar.gz ({file_size / 1024:.2f} KB)")
        
        # Extract tar.gz
        with tarfile.open(paths["model_tar_local"], "r:gz") as tar:
            tar.extractall(path=paths["extract_dir"])
        
        # List extracted files
        extracted_files = os.listdir(paths["extract_dir"])
        logger.info(f"Extracted files: {extracted_files}")
        
        # Verify required files exist
        required_files = ["sgd_model.joblib", "scaler.joblib"]
        missing_files = [f for f in required_files if f not in extracted_files]
        
        if missing_files:
            logger.error(f"Missing required files in tar.gz: {missing_files}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download/extract model: {str(e)}")
        return False


def load_model_and_scaler(paths):
    """Load model and scaler from extracted files"""
    model_path = os.path.join(paths["extract_dir"], "sgd_model.joblib")
    scaler_path = os.path.join(paths["extract_dir"], "scaler.joblib")
    
    try:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("✓ Loaded existing model and scaler")
        
        # Log model info
        logger.info(f"Model type: {type(clf)}")
        logger.info(f"Model classes: {clf.classes_ if hasattr(clf, 'classes_') else 'Not fitted yet'}")
        
        return clf, scaler
        
    except Exception as e:
        logger.error(f"Failed to load model/scaler: {str(e)}")
        return None, None


def load_csv_data():
    """Load the latest CSV file from csv_folder"""
    if not os.path.exists(INPUT_DATA_PATH):
        logger.error(f"Directory not found: {INPUT_DATA_PATH}")
        return None, None
    
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
        logger.info(f"Loaded CSV with shape: {data.shape}")
        
        # Show columns
        logger.info(f"Columns in CSV: {list(data.columns)}")
        
        # Check required columns
        required_cols = ["Date", "Location", "Rain Tomorrow"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(data.columns)}")
            return None, None
        
        # Prepare features and target
        X = data.drop(columns=["Date", "Location", "Rain Tomorrow"])
        y = data["Rain Tomorrow"]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts().to_dict()}")
        
        # Show sample of features
        logger.info(f"Feature columns: {list(X.columns)}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Failed to load CSV data: {str(e)}")
        return None, None


def create_updated_tar_gz(clf, scaler, paths):
    """Create updated model.tar.gz with new model and scaler"""
    try:
        # Save updated model and scaler to updated_model_dir
        model_path = os.path.join(paths["updated_model_dir"], "sgd_model.joblib")
        scaler_path = os.path.join(paths["updated_model_dir"], "scaler.joblib")
        
        joblib.dump(clf, model_path)
        logger.info(f"✓ Saved updated model to {model_path}")
        
        joblib.dump(scaler, scaler_path)
        logger.info(f"✓ Saved updated scaler to {scaler_path}")
        
        # Copy inference.py from extracted files or create new
        inference_src = os.path.join(paths["extract_dir"], "inference.py")
        inference_dst = os.path.join(paths["updated_model_dir"], "inference.py")
        
        if os.path.exists(inference_src):
            shutil.copy2(inference_src, inference_dst)
            logger.info("✓ Copied inference.py from existing model")
        else:
            logger.warning("inference.py not found in extracted files, creating basic version")
            # Create basic inference.py
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
            with open(inference_dst, "w") as f:
                f.write(inference_code)
        
        # Verify files exist before creating tar.gz
        files_to_check = [model_path, scaler_path, inference_dst]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
        
        # Create new tar.gz
        with tarfile.open(paths["updated_tar_path"], "w:gz") as tar:
            tar.add(model_path, arcname="sgd_model.joblib")
            tar.add(scaler_path, arcname="scaler.joblib")
            tar.add(inference_dst, arcname="inference.py")
        
        file_size = os.path.getsize(paths["updated_tar_path"])
        logger.info(f"✓ Created updated model.tar.gz ({file_size / 1024:.2f} KB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create updated tar.gz: {str(e)}", exc_info=True)
        return False


def upload_to_s3(s3_client, paths):
    """Upload updated model.tar.gz to S3"""
    try:
        s3_client.upload_file(paths["updated_tar_path"], S3_BUCKET, MODEL_TAR_S3_KEY)
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
    
    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=REGION)
    
    try:
        # Step 0: Create all necessary directories
        paths = clean_and_create_directories()
        logger.info(f"Working in directory: {paths['temp_dir']}")
        
        # Step 1: Download and extract model from S3
        if not download_and_extract_model(s3_client, paths):
            logger.error("Failed to download/extract model. Exiting.")
            sys.exit(1)
        
        # Step 2: Load model and scaler
        clf, scaler = load_model_and_scaler(paths)
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
        if not create_updated_tar_gz(clf, scaler, paths):
            logger.error("Failed to create updated model.tar.gz. Exiting.")
            sys.exit(1)
        
        # Step 6: Upload to S3
        if not upload_to_s3(s3_client, paths):
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
