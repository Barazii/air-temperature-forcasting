from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.fail_step import FailStep
import json
import boto3
from sagemaker.processing import ScriptProcessor


def go_go_go():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=True, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
    )

    # define the processor
    sklearnprocessor = SKLearnProcessor(
        framework_version=os.environ["SKL_VERSION"],
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"],
        command=[
            "python3",
        ],
        max_runtime_in_seconds=600,
        base_job_name="air-temperature-forecasting-sklearn-processor",
        sagemaker_session=sagemaker_session,
        env={
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
            "PREDICTION_LENGTH": os.environ["PREDICTION_LENGTH"],
        },
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=sklearnprocessor,
        display_name="process data",
        description="Designed mainly to split and transform the data.",
        inputs=[
            ProcessingInput(
                source=os.path.join(os.environ["S3_PROJECT_URI"], "dataset"),
                destination=os.path.join(os.environ["PC_BASE_DIR"], "dataset"),
                input_name="dataset",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
                s3_input=None,
                dataset_definition=None,
                app_managed=False,
            )
        ],
        outputs=[
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "train"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/train"
                ),
                output_name="train",
                s3_upload_mode="EndOfJob",
                app_managed=False,
                feature_store_output=None,
            ),
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "test"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/test"
                ),
                output_name="test",
                s3_upload_mode="EndOfJob",
                app_managed=False,
                feature_store_output=None,
            ),
        ],
        code="scripts/processing.py",
        cache_config=cache_config,
    )

    # defining the estimator
    image_uri = image_uris.retrieve(
        framework="forecasting-deepar",
        region=os.environ["AWS_REGION"],
        version="1",
        image_scope="training",
    )
    hyperparameters = {
        "epochs": "50",
        "time_freq": "H",
        "prediction_length": os.environ["PREDICTION_LENGTH"],
        "context_length": os.environ["PREDICTION_LENGTH"],
        "test_quantiles": "[0.2, 0.5, 0.9]",
    }
    estimator = Estimator(
        image_uri,
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["TRAINING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["TRAINING_JOB_INSTANCE_TYPE"],
        volume_size=25,
        max_run=36000,
        input_mode="File",
        output_path=f"{os.environ['S3_PROJECT_URI']}/training-step",
        base_job_name="air-temperature-forecasting-estimator",
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        model_channel_name="model",
    )

    # define the training step
    training_step = TrainingStep(
        name="train-model",
        estimator=estimator,
        display_name="train-forecasting-deepar-model",
        description="This step trains a pre-built deepAR model on the air temperature data.",
        inputs={
            "train": TrainingInput(
                processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="json",
            ),
            "test": TrainingInput(
                processing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="json",
            ),
        },
        cache_config=cache_config,
    )

    # define the inference model
    # create a model object out of the trained estimator
    model = Model(
        image_uri=image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=os.environ["SM_EXEC_ROLE"],
        predictor_cls=None,
        name="trained-forecasting-deepar-model",
        sagemaker_session=sagemaker_session,
        # entry_point=f"./scripts/inference_request_parser.py",
    )

    model_step = ModelStep(
        name="create-model",
        display_name="create model",
        description="This step creates a sagemaker model from the trained estimator.",
        step_args=model.create(
            instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"]
        ),
    )

    # define the transformer for evaluation
    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_type=os.environ["TRANSFORM_JOB_INSTANCE_TYPE"],
        instance_count=int(os.environ["TRANSFORM_JOB_INSTANCE_COUNT"]),
        strategy="MultiRecord",
        accept="application/jsonlines",
        assemble_with="Line",
        output_path=os.environ["S3_TRANSFORM_OUTPUT_URI"],
        base_transform_job_name="air-temperature-forecasting-transformer",
        sagemaker_session=sagemaker_session,
        env={
            "DEEPAR_INFERENCE_CONFIG": '{"num_samples": 100, "output_types": ["mean", "quantiles", "samples"], "quantiles": ["0.2","0.5","0.8"]}'
        },
    )
    transform_step = TransformStep(
        name="generate-test-predictions",
        display_name="generate test predictions",
        description="This step generates predictions on the test data using the trained model for evaluation.",
        step_args=transformer.transform(
            job_name="air-temperature-forecasting-transformer",
            data=Join(
                on="/",
                values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    "test.json",
                ],
            ),
            split_type="Line",
            join_source="None",
            content_type="application/jsonlines",
        ),
        cache_config=cache_config,
    )

    # define the evaluation step
    sklearnprocessor = SKLearnProcessor(
        framework_version=os.environ["SKL_VERSION"],
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"],
        env={
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
            "PREDICTION_LENGTH": os.environ["PREDICTION_LENGTH"],
        },
    )

    eval_report = PropertyFile(
        name="evaluation-report",
        output_name="report",
        path="report.json",
    )

    evaluation_step = ProcessingStep(
        name="evaluate-model",
        processor=sklearnprocessor,
        display_name="evaluate model",
        description="This step evaluates the predictions generated by the model in the transform job and calculates a performance metric and generates a report about it.",
        inputs=[
            ProcessingInput(
                source=Join(
                    on="/",
                    values=[os.environ["S3_TRANSFORM_OUTPUT_URI"], "test.json.out"],
                ),
                destination="/opt/ml/processing/input/predictions",
            ),
            ProcessingInput(
                source=Join(
                    on="/",
                    values=[
                        processing_step.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        "test.json",
                    ],
                ),
                destination="/opt/ml/processing/input/ground_truth",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/output",
                destination=os.environ["S3_EVALUATION_REPORT_URI"],
            ),
        ],
        code="scripts/evaluation.py",
        cache_config=cache_config,
        property_files=[eval_report],
    )

    # define the experiment tracking step
    sklearnprocessor = ScriptProcessor(
        image_uri="482497089777.dkr.ecr.eu-north-1.amazonaws.com/cometml:latest",
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_JOB_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_JOB_INSTANCE_TYPE"],
        env={
            "COMET_API_KEY": os.environ["COMET_API_KEY"],
            "COMET_PROJECT_NAME": os.environ["COMET_PROJECT_NAME"],
            "TRAINING_JOB_NAME": training_step.properties.TrainingJobName,
            "TRAIN_DATA_S3_URI": processing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            "EVALUATION_REPORT_S3_URI": os.environ["S3_EVALUATION_REPORT_URI"],
            "AWS_REGION": os.environ["AWS_REGION"],
        },
        command=["python3"]
    )

    tracking_step = ProcessingStep(
        name="track-experiment",
        processor=sklearnprocessor,
        display_name="track experiment",
        description="This step logs information obtained from the training and evaluation steps using Comet ML.",
        code="scripts/tracking.py",
        cache_config=cache_config,
        depends_on=[training_step, evaluation_step],
    )

    fail_step = FailStep(
        name="disregard-registration",
        error_message="Model's MAE is greater than MAE threshold. Model performance unacceptable.",
    )

    MAE_threshold = ParameterFloat(name="MAE_threshold", default_value=20.0)
    condition = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=eval_report,
            json_path="MAE",
        ),
        right=MAE_threshold,
    )

    condition_step = ConditionStep(
        name="check-model-performace",
        display_name="check model performance",
        description="This step compares model prediction error with a predefined threshold.",
        conditions=[condition],
        if_steps=[tracking_step],
        else_steps=[fail_step],
    )

    # build the pipeline
    pipeline = Pipeline(
        name="air-temperature-forecasting-pipeline",
        parameters=[MAE_threshold],
        steps=[
            processing_step,
            training_step,
            transform_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to forecast air temperature using deepAR.",
    )

    pipeline.start()


if __name__ == "__main__":
    go_go_go()
