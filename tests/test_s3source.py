import logging
import os
import shutil
from pathlib import Path

import pytest

from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    S3Source,
)

IS_CI = os.getenv("RUN_IN_CI") == "1"

# Get logger
_log = logging.getLogger(__name__)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI if the dataset in cos is very large."
)
def test_s3source():

    # Define the COS(s3) endpoints and buckets to pull the data from
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    cos_bucket = os.environ.get("S3_COS_BUCKET")
    cos_dir = os.environ.get("S3_COS_DIR")

    target_path = Path("./scratch/S3Source/")
    #  Clean the directory
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    if not endpoint:
        raise ValueError("Please set the S3_ENDPOINT environment variable")
    if not access_key:
        raise ValueError("Please set the S3_ACCESS_KEY environment variable")
    if not secret_key:
        raise ValueError("Please set the S3_SECRET_KEY environment variable")
    if not cos_bucket:
        raise ValueError("Please set the S3_COS_BUCKET environment variable")
    if not cos_dir:
        raise ValueError("Please set the S3_COS_DIR environment variable")

    dataset_source = S3Source(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=cos_bucket,
        key_prefix=cos_dir,
        overwrite_downloads=True,
    )

    dataset_builder = BaseEvaluationDatasetBuilder(
        name="s3_dataset",
        dataset_source=dataset_source,
        target=target_path,
        end_index=-1,
    )

    output_dir = dataset_builder.retrieve_input_dataset()
    assert output_dir is not None
    files = [
        f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))
    ]
    folders = [
        f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))
    ]

    _log.info("Number of Files: %s; Number of Folders: %s", len(files), len(folders))
    assert files or folders, f"The directory {output_dir} is empty."
