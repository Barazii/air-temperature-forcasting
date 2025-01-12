import json
from sagemaker_containers.beta.framework import worker


def model_fn(model_dir):
    pass


def input_fn(request_body, request_content_type):
    if request_content_type == "application/jsonlines":

        decoded_data = request_body.decode("utf-8")
        parsed_data = [json.loads(line) for line in decoded_data.strip().split("\n")]

        input_data = ""
        for dic in parsed_data:
            assert "start" in dic
            assert "target" in dic
            input_data += json.dumps(dic) + "\n"
            # this is how the input data should look like:
            # {
            #   "start": "2009-11-01 00:00:00",
            #   "target": [1.0, -5.0, ...],
            # },
            # {
            #   "start": "2009-11-01 00:00:00",
            #   "target": [1.0, -5.0, ...],
            # },
            # ...etc, where each line/json-object is a time series.

        return input_data


def predict_fn(input_data, model):
    return input_data


def output_fn(prediction, content_type):
    return worker.Response(prediction, mimetype="application/jsonlines")
