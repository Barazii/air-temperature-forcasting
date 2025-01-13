import pytest
import tempfile
from pathlib import Path
import boto3
from evaluation import evaluate
import os


@pytest.fixture(scope="function", autouse=False)
def directory(monkeypatch):
    directory = Path(tempfile.mkdtemp())
    ground_truth_dir = directory / "input" / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    bucket_name = "airtemperatureforecasting"
    dataset_folder_s3_prefix = "processing-step/test"
    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=dataset_folder_s3_prefix)[
        "Contents"
    ]:
        s3.download_file(
            bucket_name, obj["Key"], str(ground_truth_dir / Path(obj["Key"]).name)
        )

    predictions_dir = directory / "input" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    dataset_folder_s3_prefix = "transform"
    obj = s3.list_objects_v2(Bucket=bucket_name, Prefix=dataset_folder_s3_prefix)[
        "Contents"
    ][0]
    s3.download_file(
        bucket_name, obj["Key"], str(predictions_dir / Path(obj["Key"]).name)
    )

    # set the environment variables
    monkeypatch.setenv("PREDICTION_LENGTH", 336)

    evaluate(directory)

    return directory


def test_evaluation_step(directory):
    assert "input" in os.listdir(directory)
    assert "output" in os.listdir(directory)
    assert "predictions" in os.listdir(directory / "input")
    assert "ground_truth" in os.listdir(directory / "input")
    assert "report.json" in os.listdir(directory / "output")

    file = os.listdir(directory / "input" / "predictions")[0]
    assert file.endswith(".json.out")

    file = os.listdir(directory / "input" / "ground_truth")[0]
    assert file.endswith(".json")
