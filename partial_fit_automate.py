import os
import pandas as pd
import joblib
import boto3
import tarfile
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import sagemaker
from sagemaker.model import Model


# ================
# CONFIG VARIABLES
# ================
S3_BUCKET = os.environ.get("S3_BUCKET", "test-rain-predictor")             # e.g. 'my-bucket'
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "test-rain-predictor")     # e.g. 'weather-endpoint'
REGION = os.environ.get("AWS_REGION", "us-east-1")  
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::493272324412:role/service-role/AmazonSageMaker-ExecutionRole-20251128T174447")   

# folder inside your GitHub repo containing CSV files
input_data_path = "csv_folder"

# temporary directory inside CodeBuild
model_dir = "/tmp/model"
os.makedirs(model_dir, exist_ok=True)


# ============================
# 1. LOAD NEW CSV FROM REPO
# ============================
csv_files = [f for f in os.listdir(input_data_path) if f.endswith(".csv")]

if not csv_files:
    raise Exception("❌ No CSV file found in csv_folder/")

# Use the latest CSV
csv_path = os.path.join(input_data_path, sorted(csv_files)[-1])
print(f"Using CSV: {csv_path}")

data = pd.read_csv(csv_path)

X = data.drop(columns=["Date", "Location", "Rain Tomorrow"])
y = data["Rain Tomorrow"]


# ============================
# 2. DOWNLOAD EXISTING MODEL FROM S3
# ============================
s3 = boto3.client("s3", region_name=REGION)

model_s3_key = "model/sgd_model.joblib"
scaler_s3_key = "model/scaler.joblib"

model_path = os.path.join(model_dir, "sgd_model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")

try:
    s3.download_file(S3_BUCKET, model_s3_key, model_path)
    s3.download_file(S3_BUCKET, scaler_s3_key, scaler_path)
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Loaded existing model & scaler from S3")
except Exception:
    print("No existing model found, creating new one...")
    clf = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    clf.partial_fit(X_scaled, y, classes=[0, 1])

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    s3.upload_file(model_path, S3_BUCKET, model_s3_key)
    s3.upload_file(scaler_path, S3_BUCKET, scaler_s3_key)

    print("Initial model created and uploaded to S3")
    exit(0)


# ============================
# 3. APPLY PARTIAL FIT
# ============================
X_scaled = scaler.transform(X)
clf.partial_fit(X_scaled, y)

# save local updates
joblib.dump(clf, model_path)
joblib.dump(scaler, scaler_path)

# upload updated model files
s3.upload_file(model_path, S3_BUCKET, model_s3_key)
s3.upload_file(scaler_path, S3_BUCKET, scaler_s3_key)

print("Updated model & scaler uploaded to S3")


# ============================
# 4. PACKAGE MODEL INTO model.tar.gz
# ============================
tar_path = "/tmp/model.tar.gz"

with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(model_path, arcname="sgd_model.joblib")
    tar.add(scaler_path, arcname="scaler.joblib")
    tar.add("inference.py", arcname="inference.py")   # very important

model_tar_s3_key = "model/model.tar.gz"
s3.upload_file(tar_path, S3_BUCKET, model_tar_s3_key)

print("model.tar.gz created & uploaded to S3")


# ============================
# 5. UPDATE EXISTING ENDPOINT
# ============================
session = sagemaker.Session(
    boto_session=boto3.Session(region_name=REGION)
)

sm_model = Model(
    model_data=f"s3://{S3_BUCKET}/{model_tar_s3_key}",
    role=SAGEMAKER_ROLE,
    sagemaker_session=session,
    entry_point="inference.py"
)

print(f"Updating endpoint: {ENDPOINT_NAME} ...")

sm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=ENDPOINT_NAME,
    update_endpoint=True
)

print("\n✅ Partial fit complete!")
print(f"➡️ Endpoint '{ENDPOINT_NAME}' updated successfully.")
