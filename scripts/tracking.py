from comet_ml import Experiment
import boto3
import os
import json


class ExperimentTracker(Experiment):
    def __init__(self, api_key, project_name):
        super().__init__(api_key=api_key, project_name=project_name)
        self.s3_client = boto3.client("s3")
        self.sm_client = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])

    def log_hyperparameters(self):
        response = self.sm_client.describe_training_job(
            TrainingJobName=os.environ["TRAINING_JOB_NAME"]
        )
        HyperParameters = response["HyperParameters"]
        self.log_parameters(HyperParameters)

    def log_dataset(self):
        train_data_s3_uri = os.environ["TRAIN_DATA_S3_URI"]
        train_data_bucket, train_data_key = train_data_s3_uri.replace(
            "s3://", ""
        ).split("/", 1)
        self.s3_client.download_file(
            train_data_bucket,
            os.path.join(train_data_key, "train.json"),
            "/tmp/train.json",
        )
        data = []
        with open("/tmp/train.json", "rb") as f:
            for line in f:
                data.append(json.loads(line.decode("utf-8")))
        self.log_dataset_hash(data)

    def log_mae(self):
        evaluation_report_s3_uri = os.environ["EVALUATION_REPORT_S3_URI"]
        evaluation_report_bucket, evaluation_report_key = (
            evaluation_report_s3_uri.replace("s3://", "").split("/", 1)
        )
        self.s3_client.download_file(
            evaluation_report_bucket,
            os.path.join(evaluation_report_key, "report.json"),
            "/tmp/report.json",
        )
        with open("/tmp/report.json", "r") as f:
            report = json.load(f)
        self.log_metrics({"MAE": report["MAE"]})

    def log_model(self):
        response = self.sm_client.describe_training_job(
            TrainingJobName=os.environ["TRAINING_JOB_NAME"]
        )
        model_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        bucket, key = model_s3_uri.replace("s3://", "").split("/", 1)
        self.s3_client.download_file(bucket, key, "/tmp/model.tar.gz")
        super().log_model("deepar_model", "/tmp/model.tar.gz")


if __name__ == "__main__":
    et = ExperimentTracker(
        api_key=os.environ["COMET_API_KEY"],
        project_name=os.environ["COMET_PROJECT_NAME"],
    )
    et.log_hyperparameters()
    et.log_dataset()
    et.log_mae()
    et.log_model()
