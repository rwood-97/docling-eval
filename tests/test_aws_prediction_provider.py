import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types import DoclingDocument

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.prediction_providers.aws_prediction_provider import AWSTextractPredictionProvider

IS_CI = bool(os.getenv("CI"))

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


def test_aws_prediction_to_docling_converter(mocker):
    os.environ['AWS_ACCESS_KEY_ID'] = "1234"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "4567"
    mocked_constructor = mocker.patch("PIL.Image")
    mocked_instance = MagicMock()
    mocked_instance.return_value.size = (100,100)
    mocked_constructor.return_value = mocked_instance
    with open('/Users/samved/Documents/github/ds4sd/docling-eval/tests/data/files/picture_classification.pdf', 'rb') as file:
        file_bytes = file.read()
    dataset_record = DatasetRecord(document_id="123", GroundTruthDocument=DoclingDocument(name="123"), GroundTruthPageImages=[mocked_instance])
    prediction_provider = AWSTextractPredictionProvider()
    doc, seg_pages = prediction_provider.convert_aws_output_to_docling(dict(), dataset_record,file_bytes)
