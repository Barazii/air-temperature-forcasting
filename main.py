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
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.pipeline import PipelineModel


def go_go_go():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=True, expire_after="3d")

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

    # build the pipeline
    pipeline = Pipeline(
        name="air-temperature-forecasting-pipeline",
        parameters=None,
        steps=[
            processing_step,
            training_step,
            transform_step,
            # condition_step,
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
