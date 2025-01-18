import pytest
from tracking import ExperimentTracker


@pytest.fixture(scope="function", autouse=False)
def et(monkeypatch):
    # set mock-up environment variables
    monkeypatch.setenv(
        "TRAINING_JOB_NAME", "pipelines-4new6tehs2bu-train-model-M8fiyKLYou"
    )
    monkeypatch.setenv("AWS_REGION", "eu-north-1")
    monkeypatch.setenv(
        "TRAIN_DATA_S3_URI",
        "s3://airtemperatureforecasting/processing-step/train",
    )
    monkeypatch.setenv(
        "EVALUATION_REPORT_S3_URI",
        "s3://airtemperatureforecasting/evaluation-step",
    )
    monkeypatch.setenv("COMET_API_KEY", "R59U4u9W6DR7Wvz860dFVymom")
    monkeypatch.setenv("COMET_PROJECT_NAME", "air-temperature-forecasting")

    api_key = "R59U4u9W6DR7Wvz860dFVymom"
    project_name = "dummy-project-for-testing"
    et = ExperimentTracker(api_key=api_key, project_name=project_name)

    return et


def test_et(et):
    et.log_hyperparameters()
    et.log_dataset()
    et.log_mae()
    et.log_model()
