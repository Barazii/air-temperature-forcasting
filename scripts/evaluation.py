import json
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import os

PREDICTION_LENGTH = 336


def evaluate(pc_base_dir):
    # Load predictions
    predictions_dir = pc_base_dir / "input" / "predictions"
    with open(predictions_dir / "test.json.out", "r") as f:
        predictions = [json.loads(line) for line in f]

    # Load ground truth
    ground_truth_dir = pc_base_dir / "input" / "ground_truth"
    with open(ground_truth_dir / "test.json", "r") as f:
        ground_truth = [json.loads(line) for line in f]

    # Extract forecast means and actuals
    forecast_means = [p["mean"] for p in predictions]
    actuals = [
        g["target"][len(g["target"]) - PREDICTION_LENGTH :] for g in ground_truth
    ]
    assert len(forecast_means) == len(
        actuals
    ), "mismatch in the number of time series between prediction file and ground truth file."
    for i in range(len(forecast_means)):
        assert (
            len(actuals[i]) == len(forecast_means[i]) == PREDICTION_LENGTH
        ), "mismatch in the length of the data arrays/vectors between predicted data and ground truth data."

    # Calculate error metrics
    mae = mean_absolute_error(actuals, forecast_means)

    # Save metrics
    report_dir = pc_base_dir / "output"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "report.json", "w") as f:
        json.dump(
            {
                "MAE": mae,
            },
            f,
        )


if __name__ == "__main__":
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    assert pc_base_dir is not None
    evaluate(pc_base_dir)
