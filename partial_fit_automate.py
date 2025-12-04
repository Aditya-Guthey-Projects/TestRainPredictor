import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
import tempfile
import shutil
from datetime import datetime
import subprocess
import sys
import traceback
import tarfile
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
region = "ap-south-1"
bucket = "rain-prediction-autodeploy-mumbai"
role = "arn:aws:iam::493272324412:role/service-role/AmazonSageMaker-ExecutionRole-20251203T141656"
endpoint_name = 'test-train-rain-auto-mumbai-1'

# Folder containing CSV files
CSV_FOLDER = "csv_folder"

# Initialize AWS clients
s3 = boto3.client('s3', region_name=region)
sagemaker_session = Session(boto_session=boto3.Session(region_name=region))
sagemaker_client = boto3.client('sagemaker', region_name=region)

def extract_existing_model():
    """Extract existing model.tar.gz from /tmp."""
    extracted_dir = "/tmp/extracted_model"
    
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)
    
    model_tar_path = "/tmp/model.tar.gz"
    
    if os.path.exists(model_tar_path):
        print(f"Extracting existing model from {model_tar_path}")
        try:
            with tarfile.open(model_tar_path, 'r:gz') as tar:
                tar.extractall(path=extracted_dir)
            
            print(f"Extracted files in {extracted_dir}:")
            for file in os.listdir(extracted_dir):
                print(f"  - {file}")
            
            return extracted_dir
        except Exception as e:
            print(f"Error extracting model: {e}")
            return None
    else:
        print("No existing model.tar.gz found")
        return None

def load_all_csvs_from_folder():
    """Load all CSV files from the csv_folder."""
    print(f"Loading CSV files from {CSV_FOLDER}...")
    
    if not os.path.exists(CSV_FOLDER):
        print(f"Error: Folder '{CSV_FOLDER}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Directory contents:")
        for item in os.listdir('.'):
            print(f"  - {item}")
        raise FileNotFoundError(f"Folder '{CSV_FOLDER}' not found")
    
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_FOLDER}")
    
    print(f"Found {len(csv_files)} CSV files:")
    data_frames = []
    
    for csv_file in csv_files:
        try:
            print(f"  Loading {os.path.basename(csv_file)}...")
            df = pd.read_csv(csv_file)
            
            # Check required columns
            required_cols = ['Temperature', 'Humidity', 'Wind Speed', 
                             'Precipitation', 'Cloud Cover', 'Pressure', 
                             'Rain Tomorrow']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  Warning: {os.path.basename(csv_file)} missing columns: {missing_cols}")
                # Try to find similar columns
                for col in missing_cols[:]:
                    for actual_col in df.columns:
                        if col.lower() in actual_col.lower():
                            df = df.rename(columns={actual_col: col})
                            missing_cols.remove(col)
                            print(f"    Renamed {actual_col} to {col}")
                            break
            
            if not missing_cols:
                data_frames.append(df)
                print(f"    Added with shape {df.shape}")
            else:
                print(f"    Skipping - still missing: {missing_cols}")
                
        except Exception as e:
            print(f"    Error loading {csv_file}: {e}")
    
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        print(f"Combined {len(data_frames)} CSV files, total shape: {combined_df.shape}")
        return combined_df
    else:
        raise Exception(f"No valid CSV files found in {CSV_FOLDER}")

def load_existing_model(extracted_dir):
    """Load existing model from extracted directory."""
    if extracted_dir and os.path.exists(extracted_dir):
        model_path = os.path.join(extracted_dir, "model.joblib")
        metadata_path = os.path.join(extracted_dir, "metadata.json")
        
        model = None
        metadata = {}
        
        # Load model if exists
        if os.path.exists(model_path):
            try:
                print(f"Loading existing model from {model_path}")
                model = joblib.load(model_path)
                print(f"Loaded model: {type(model).__name__}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Load metadata if exists
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: {metadata.get('model_type', 'Unknown')}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        else:
            metadata = create_default_metadata()
        
        return model, metadata
    else:
        print("No extracted directory found, creating new model")
        return None, create_default_metadata()

def create_default_metadata():
    """Create default metadata for new model."""
    return {
        "model_type": "SGDClassifier",
        "features": ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure'],
        "target": "Rain Tomorrow",
        "created_at": datetime.now().isoformat(),
        "update_count": 0,
        "last_update": datetime.now().isoformat(),
        "feature_stats": {},
        "description": "Rain prediction model using SGDClassifier",
        "training_params": {
            "loss": "log_loss",
            "penalty": "l2",
            "learning_rate": "optimal",
            "eta0": 0.01,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": 42
        }
    }

def preprocess_weather_data(df):
    """Preprocess weather data for training."""
    print("\nPreprocessing weather data...")
    
    # Keep only relevant columns
    features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']
    target = 'Rain Tomorrow'
    
    # Drop unnecessary columns
    cols_to_drop = ['Date', 'Location', 'date', 'location']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")
    
    # Ensure all required columns exist
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        # Try case-insensitive matching
        for missing in missing_features[:]:
            for col in df.columns:
                if missing.lower() == col.lower():
                    df = df.rename(columns={col: missing})
                    missing_features.remove(missing)
                    print(f"  Renamed {col} to {missing}")
                    break
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    
    # Separate features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle missing values
    if X.isnull().any().any():
        missing_count = X.isnull().sum().sum()
        print(f"Filling {missing_count} missing values...")
        X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # Ensure correct data types
    X = X.astype(np.float32)
    
    # Convert target to binary (0 or 1)
    y = pd.to_numeric(y, errors='coerce').fillna(0)
    y = (y > 0).astype(np.int32)  # Convert to binary
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Print target distribution
    target_counts = y.value_counts()
    print(f"Target distribution:")
    print(f"  No Rain (0): {target_counts.get(0, 0)} samples")
    print(f"  Rain (1): {target_counts.get(1, 0)} samples")
    
    return X, y

def train_or_update_model(X, y, existing_model, metadata):
    """Train new model or update existing one using SGDClassifier."""
    print("\n" + "="*50)
    print("Training/Updating Model with SGDClassifier")
    print("="*50)
    
    # Install scikit-learn 1.7.2
    print("Ensuring scikit-learn 1.7.2 is installed...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.7.2", "-q"])
    except:
        pass
    
    from sklearn.linear_model import SGDClassifier
    
    # Convert to numpy arrays
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    # Check if we have an existing SGDClassifier
    if existing_model is not None and isinstance(existing_model, SGDClassifier):
        print("Found existing SGDClassifier model")
        
        if hasattr(existing_model, 'partial_fit'):
            print("Performing partial_fit...")
            
            # Get unique classes
            unique_classes = np.unique(y_array)
            print(f"Classes in new data: {unique_classes}")
            
            try:
                # Perform partial fit
                existing_model.partial_fit(X_array, y_array, classes=unique_classes)
                print("Partial fit completed successfully!")
                model = existing_model
            except Exception as e:
                print(f"Partial fit failed: {e}")
                print("Training new SGDClassifier from scratch...")
                model = train_new_sgdclassifier(X_array, y_array, metadata)
        else:
            print("Existing model doesn't support partial_fit, training new model...")
            model = train_new_sgdclassifier(X_array, y_array, metadata)
    else:
        print("Training new SGDClassifier from scratch...")
        model = train_new_sgdclassifier(X_array, y_array, metadata)
    
    return model, metadata

def train_new_sgdclassifier(X_array, y_array, metadata):
    """Train a new SGDClassifier from scratch."""
    from sklearn.linear_model import SGDClassifier
    
    print("Training new SGDClassifier...")
    
    # Get training parameters from metadata or use defaults
    training_params = metadata.get('training_params', {
        "loss": "log_loss",
        "penalty": "l2",
        "learning_rate": "optimal",
        "eta0": 0.01,
        "max_iter": 1000,
        "tol": 1e-3,
        "random_state": 42
    })
    
    # Create SGDClassifier with specified parameters
    model = SGDClassifier(
        loss=training_params["loss"],
        penalty=training_params["penalty"],
        learning_rate=training_params["learning_rate"],
        eta0=training_params.get("eta0", 0.01),
        max_iter=training_params.get("max_iter", 1000),
        tol=training_params.get("tol", 1e-3),
        random_state=training_params.get("random_state", 42),
        verbose=1  # Add verbose output
    )
    
    print(f"SGDClassifier parameters:")
    print(f"  loss: {training_params['loss']}")
    print(f"  penalty: {training_params['penalty']}")
    print(f"  learning_rate: {training_params['learning_rate']}")
    print(f"  max_iter: {training_params.get('max_iter', 1000)}")
    
    # Fit the model
    model.fit(X_array, y_array)
    print(f"Model trained successfully!")
    
    return model

def evaluate_model(model, X, y):
    """Evaluate model performance."""
    try:
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        y_pred = model.predict(X_array)
        
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        
        results = {
            "accuracy": float(accuracy_score(y_array, y_pred)),
            "samples": len(y_array),
            "positive_samples": int(y_array.sum()),
            "negative_samples": int(len(y_array) - y_array.sum())
        }
        
        # Classification report
        if len(np.unique(y_array)) <= 10:
            report = classification_report(y_array, y_pred, output_dict=True)
            results.update({
                "precision": float(report['weighted avg']['precision']),
                "recall": float(report['weighted avg']['recall']),
                "f1_score": float(report['weighted avg']['f1-score'])
            })
        
        # ROC AUC if available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_array)[:, 1]
                results["roc_auc"] = float(roc_auc_score(y_array, y_proba))
            except:
                pass
        
        print(f"\nModel Evaluation:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Samples: {results['samples']}")
        print(f"  Rain (1): {results['positive_samples']}, No Rain (0): {results['negative_samples']}")
        
        if 'precision' in results:
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

def create_model_package(model, metadata, X=None):
    """Create model.tar.gz package."""
    print("\nCreating model.tar.gz package...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Working directory: {temp_dir}")
    
    try:
        # Update metadata
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['update_count'] = metadata.get('update_count', 0) + 1
        metadata['model_type'] = type(model).__name__
        metadata['sklearn_version'] = '1.7.2'
        
        # Add SGDClassifier parameters
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            # Filter to keep only relevant parameters
            relevant_params = ['loss', 'penalty', 'learning_rate', 'eta0', 'max_iter', 'tol', 'random_state']
            metadata['model_params'] = {k: model_params.get(k) for k in relevant_params if k in model_params}
        
        # Add feature statistics if X is provided
        if X is not None:
            metadata['feature_stats'] = {
                'means': {col: float(X[col].mean()) for col in X.columns},
                'stds': {col: float(X[col].std()) for col in X.columns},
                'mins': {col: float(X[col].min()) for col in X.columns},
                'maxs': {col: float(X[col].max()) for col in X.columns}
            }
        
        # 1. Save model
        model_path = os.path.join(temp_dir, "model.joblib")
        joblib.dump(model, model_path, protocol=4)
        print(f"✓ Model saved: {model_path}")
        
        # 2. Save metadata
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")
        
        # 3. Create requirements.txt
        req_path = os.path.join(temp_dir, "requirements.txt")
        with open(req_path, 'w') as f:
            f.write("""scikit-learn==1.7.2
joblib==1.3.2
numpy==1.24.3
pandas==2.0.3""")
        print(f"✓ requirements.txt created")
        
        # 4. Create inference.py
        inference_path = os.path.join(temp_dir, "inference.py")
        create_inference_script(inference_path, metadata)
        print(f"✓ inference.py created")
        
        # 5. Create tar.gz
        output_tar = "/tmp/new_model.tar.gz"
        with tarfile.open(output_tar, 'w:gz') as tar:
            for file_name in ["model.joblib", "metadata.json", "requirements.txt", "inference.py"]:
                file_path = os.path.join(temp_dir, file_name)
                tar.add(file_path, arcname=file_name)
        
        print(f"✓ Created model.tar.gz: {output_tar}")
        print(f"✓ File size: {os.path.getsize(output_tar) / 1024:.2f} KB")
        
        return output_tar
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary directory")

def create_inference_script(file_path, metadata):
    """Create inference.py script for SGDClassifier."""
    script_content = """import os
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
   
    print(f"Output content type: {content_type}")
    return json.dumps(prediction)
"""
    
    with open(file_path, 'w') as f:
        f.write(script_content)

def upload_to_s3_and_deploy(tar_path, metadata):
    """Upload model to S3 and deploy to SageMaker."""
    print("\n" + "="*50)
    print("Uploading to S3 and Deploying")
    print("="*50)
    
    try:
        # 1. Upload to S3 with the same name
        s3_model_key = "model/model.tar.gz"
        print(f"Uploading to s3://{bucket}/{s3_model_key}")
        
        s3.upload_file(
            Filename=tar_path,
            Bucket=bucket,
            Key=s3_model_key
        )
        
        s3_url = f"s3://{bucket}/{s3_model_key}"
        print(f"✓ Model uploaded to S3: {s3_url}")
        
        # 2. Upload metadata separately
        metadata_key = "model/metadata.json"
        metadata_json = json.dumps(metadata, indent=2)
        s3.put_object(Bucket=bucket, Key=metadata_key, Body=metadata_json)
        print(f"✓ Metadata uploaded to S3")
        
        # 3. Deploy to SageMaker
        print(f"\nDeploying to SageMaker endpoint: {endpoint_name}")
        
        sklearn_model = SKLearnModel(
            model_data=s3_url,
            role=role,
            entry_point='inference.py',
            framework_version='1.4-2',
            py_version='py3',
            sagemaker_session=sagemaker_session,
            name=f"sgd-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Check if endpoint exists
        try:
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            print(f"Found existing endpoint: {endpoint_info['EndpointStatus']}")
            
            if endpoint_info['EndpointStatus'] == 'InService':
                print("Updating existing endpoint...")
                predictor = sklearn_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.xlarge',
                    endpoint_name=endpoint_name,
                    update_endpoint=True,
                    wait=True
                )
                print("✓ Endpoint updated successfully!")
            else:
                print(f"Endpoint status is {endpoint_info['EndpointStatus']}, creating new...")
                predictor = sklearn_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.xlarge',
                    endpoint_name=endpoint_name,
                    wait=True
                )
                print("✓ New endpoint created successfully!")
                
        except sagemaker_client.exceptions.ClientError as e:
            if 'Could not find endpoint' in str(e):
                print("Endpoint doesn't exist, creating new...")
                predictor = sklearn_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.xlarge',
                    endpoint_name=endpoint_name,
                    wait=True
                )
                print("✓ New endpoint created successfully!")
            else:
                raise
        
        return predictor, s3_url
        
    except Exception as e:
        print(f"❌ Error in upload/deploy: {str(e)}")
        traceback.print_exc()
        raise

def test_endpoint():
    """Test the deployed endpoint."""
    print("\n" + "="*50)
    print("Testing Endpoint")
    print("="*50)
    
    runtime = boto3.client('runtime.sagemaker', region_name=region)
    
    test_cases = [
        {
            "name": "Batch prediction",
            "data": {
                "instances": [
                    [87.5, 75.7, 28.4, 0.0, 69.6, 1026.0],    # From your data
                    [83.3, 28.7, 12.4, 0.5, 41.6, 995.9],     # From your data
                    [80.9, 64.7, 14.2, 0.9, 77.4, 980.8]      # From your data
                ]
            }
        },
        {
            "name": "Single prediction with dict",
            "data": {
                "Temperature": 78.1,
                "Humidity": 59.7,
                "Wind Speed": 19.4,
                "Precipitation": 0.09,
                "Cloud Cover": 52.5,
                "Pressure": 979.0
            }
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test['data'])
            )
            
            result = json.loads(response['Body'].read().decode())
            
            if result.get('status') == 'success':
                print(f"✓ Success!")
                print(f"  Predictions: {result.get('predictions', [])}")
                if 'rain_probability' in result:
                    print(f"  Rain Probabilities: {result['rain_probability']}")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")

def main():
    """Main execution function."""
    print("="*60)
    print("SGDClassifier MODEL UPDATE PIPELINE")
    print("="*60)
    print(f"Region: {region}")
    print(f"S3 Bucket: {bucket}")
    print(f"Endpoint: {endpoint_name}")
    print(f"CSV Folder: {CSV_FOLDER}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    summary = {
        "status": "unknown",
        "timestamp": datetime.now().isoformat(),
        "endpoint_name": endpoint_name,
        "model_type": "SGDClassifier"
    }
    
    try:
        # Step 1: Extract existing model
        print("\n1. EXTRACTING EXISTING MODEL")
        extracted_dir = extract_existing_model()
        
        # Step 2: Load existing model and metadata
        print("\n2. LOADING EXISTING MODEL")
        existing_model, metadata = load_existing_model(extracted_dir)
        
        # Step 3: Load CSV files from folder
        print("\n3. LOADING DATA FROM CSV_FOLDER")
        new_data = load_all_csvs_from_folder()
        summary["csv_files_loaded"] = len(new_data) if hasattr(new_data, '__len__') else 1
        
        # Step 4: Preprocess data
        print("\n4. PREPROCESSING DATA")
        X, y = preprocess_weather_data(new_data)
        summary["training_samples"] = len(X)
        summary["positive_samples"] = int(y.sum())
        
        # Step 5: Train/Update model with SGDClassifier
        print("\n5. TRAINING SGDCLASSIFIER MODEL")
        model, metadata = train_or_update_model(X, y, existing_model, metadata)
        summary["model_type"] = type(model).__name__
        summary["update_count"] = metadata.get('update_count', 1)
        
        # Step 6: Evaluate model
        print("\n6. EVALUATING MODEL")
        eval_results = evaluate_model(model, X, y)
        if eval_results:
            metadata['last_evaluation'] = eval_results
            summary["evaluation"] = eval_results
        
        # Step 7: Create model package
        print("\n7. CREATING MODEL PACKAGE")
        tar_path = create_model_package(model, metadata, X)
        
        # Step 8: Upload and deploy
        print("\n8. UPLOADING AND DEPLOYING")
        predictor, s3_url = upload_to_s3_and_deploy(tar_path, metadata)
        summary["s3_model_url"] = s3_url
        summary["endpoint_arn"] = predictor.endpoint_name if hasattr(predictor, 'endpoint_name') else endpoint_name
        
        # Step 9: Test endpoint
        print("\n9. TESTING ENDPOINT")
        test_endpoint()
        
        # Success summary
        summary["status"] = "success"
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model: {summary['model_type']}")
        print(f"Accuracy: {eval_results.get('accuracy', 'N/A') if eval_results else 'N/A':.4f}")
        print(f"Samples: {summary['training_samples']}")
        print(f"Update Count: {summary['update_count']}")
        print(f"S3: {summary['s3_model_url']}")
        print(f"Endpoint: {summary['endpoint_arn']}")
        print("="*60)
        
    except Exception as e:
        summary["status"] = "failed"
        summary["error"] = str(e)
        summary["traceback"] = traceback.format_exc()
        
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        traceback.print_exc()
    
    # Save summary
    summary_path = "/tmp/run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))
    
    return summary["status"] == "success"

if __name__ == "__main__":
    # Install required packages
    try:
        import sklearn
        import joblib
        import pandas as pd
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "scikit-learn==1.7.2", "joblib==1.3.2", 
                               "pandas==2.0.3", "boto3", "sagemaker", "-q"])
    
    success = main()
    sys.exit(0 if success else 1)
