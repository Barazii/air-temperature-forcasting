from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from dotenv import load_dotenv
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os


def go_go_go():
    load_dotenv()

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
        env={"PC_BASE_DIR": os.environ["PC_BASE_DIR"]},
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=sklearnprocessor,
        display_name="process data",
        description="First step of the pipeline. Designed mainly to split and transform the data.",
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
        cache_config=None,
    )

    # build the pipeline
    pipeline = Pipeline(
        name="air-temperature-forecasting-pipeline",
        parameters=None,
        steps=[
            processing_step,
            # tuning_step,
            # evaluation_step,
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
