import os
import pytest
import tempfile
from pathlib import Path
import boto3
import json
from processing import processing


@pytest.fixture(scope="function", autouse=False)
def directory(monkeypatch):
    directory = Path(tempfile.mkdtemp())
    dataset_directory = directory / "dataset"
    dataset_directory.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    bucket_name = "airtemperatureforecasting"
    dataset_folder_s3_prefix = "dataset"
    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=dataset_folder_s3_prefix)[
        "Contents"
    ]:
        s3.download_file(
            bucket_name, obj["Key"], str(dataset_directory / Path(obj["Key"]).name)
        )

    # set the environment variables
    monkeypatch.setenv("PREDICTION_LENGTH", 336)

    processing(directory)

    return directory


def test_processing_step_generates_the_right_dirs_and_contents(directory):
    assert "dataset" in os.listdir(directory)
    assert "train" in os.listdir(directory)
    assert "test" in os.listdir(directory)
    assert len(os.listdir(directory / "dataset")) == 12
    assert len(os.listdir(directory / "train")) == 1
    assert len(os.listdir(directory / "test")) == 1

    data = []
    with open(directory / "train/train_data.json", "rb") as f:
        for line in f:
            dic = json.loads(line.decode("utf-8"))
            data.append(dic)

    for dic in data:
        assert isinstance(dic["start"], str)
        assert isinstance(dic["target"], list)
        assert dic.keys() == {"start", "target"}

    data = []
    with open(directory / "test/test_data.json", "rb") as f:
        for line in f:
            dic = json.loads(line.decode("utf-8"))
            data.append(dic)

    for dic in data:
        assert isinstance(dic["start"], str)
        assert isinstance(dic["target"], list)
        assert dic.keys() == {"start", "target"}
