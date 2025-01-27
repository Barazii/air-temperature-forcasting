import os
from pathlib import Path
import pandas as pd
import json


def _save_data_splits(train_data, test_data, pc_base_dir):
    # train split
    path = pc_base_dir / "train"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "train.json", "wb") as f:
        for dic in train_data:
            f.write(json.dumps(dic, separators=(",", ":")).encode("utf-8"))
            f.write("\n".encode("utf-8"))

    # test split
    path = pc_base_dir / "test"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "test.json", "wb") as f:
        for dic in test_data:
            f.write(json.dumps(dic, separators=(",", ":")).encode("utf-8"))
            f.write("\n".encode("utf-8"))


def _read_csv_data(pc_base_dir):
    paths = (pc_base_dir / "dataset").glob("*.csv")
    datasets = [pd.read_csv(p) for p in paths]
    return datasets


def processing(pc_base_dir):
    datasets = _read_csv_data(pc_base_dir)

    train_data = [
        {
            "start": str(df["datetime"].min()),
            "target": df["TEMP"]
            .fillna(df["TEMP"].mean())
            .tolist()[: -int(os.environ["PREDICTION_LENGTH"])],
        }
        for df in datasets
    ]

    test_data = [
        item
        for df in datasets
        for item in [
            {
                "start": str(df["datetime"].min()),
                "target": df["TEMP"].fillna(df["TEMP"].mean()).tolist(),
            },
            {
                "start": str(df["datetime"].min()),
                "target": df["TEMP"]
                .fillna(df["TEMP"].mean())
                .tolist()[: -int(os.environ["PREDICTION_LENGTH"])],
            },
            {
                "start": str(df["datetime"].min()),
                "target": df["TEMP"]
                .fillna(df["TEMP"].mean())
                .tolist()[: -2 * int(os.environ["PREDICTION_LENGTH"])],
            },
        ]
    ]

    _save_data_splits(train_data, test_data, pc_base_dir)


if __name__ == "__main__":
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    assert pc_base_dir is not None
    processing(pc_base_dir)
