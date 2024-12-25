from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from dotenv import load_dotenv
import os


def go():
    load_dotenv()

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET"],
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
        base_job_name="air-temperature-forecasting-processing-job",
        sagemaker_session=sagemaker_session,
        env=None,
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=sklearnprocessor,
        display_name="process data",
        description="First step of the pipeline. Designed mainly to split and transform the data.",
        inputs=[],
        outputs=[],
        job_arguments=["--pc_base_dir", os.environ["PC_BASE_DIR"]],
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


if __name__ == "__main__":
    go()
