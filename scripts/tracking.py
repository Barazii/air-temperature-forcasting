from comet_ml import Experiment
import boto3
import os
import json

experiment = Experiment(
    api_key=os.environ["COMET_API_KEY"], project_name=os.environ["COMET_PROJECT_NAME"]
)

# Get training job details
client = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])
response = client.describe_training_job(TrainingJobName=os.environ["TRAINING_JOB_NAME"])

# Log hyperparameters
HyperParameters = response["HyperParameters"]
experiment.log_parameters(HyperParameters)

# Log dataset hash
train_data_s3_uri = os.environ["TRAIN_DATA_S3_URI"]
train_data_bucket, train_data_key = train_data_s3_uri.replace("s3://", "").split("/", 1)
s3 = boto3.client("s3")
s3.download_file(
    train_data_bucket, os.path.join(train_data_key, "train.json"), "/tmp/train.json"
)
data = []
with open("/tmp/train.json", "rb") as f:
    for line in f:
        data.append(json.loads(line.decode("utf-8")))
experiment.log_dataset_hash(data)

# Log mae
evaluation_report_s3_uri = os.environ["EVALUATION_REPORT_S3_URI"]
evaluation_report_bucket, evaluation_report_key = evaluation_report_s3_uri.replace(
    "s3://", ""
).split("/", 1)
s3.download_file(
    evaluation_report_bucket,
    os.path.join(evaluation_report_key, "report.json"),
    "/tmp/report.json",
)
with open("/tmp/report.json", "r") as f:
    report = json.load(f)
experiment.log_metrics({"MAE": report["MAE"]})

# Log model artifact
model_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
bucket, key = model_s3_uri.replace("s3://", "").split("/", 1)
s3 = boto3.client("s3")
s3.download_file(bucket, key, "/tmp/model.tar.gz")
experiment.log_model("deepar_model", "/tmp/model.tar.gz")
