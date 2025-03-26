import copy
from typing import Dict, Optional

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter, FormatOption

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import docling_version


class DoclingPredictionProvider(BasePredictionProvider):
    def __init__(
        self,
        format_options: Optional[Dict[InputFormat, FormatOption]] = None,
        do_visualization: bool = False,
    ):
        super().__init__(do_visualization=do_visualization)
        self.doc_converter = DocumentConverter(format_options=format_options)

    @property
    def prediction_format(self) -> PredictionFormats:
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(
        self,
        record: DatasetRecord,
    ) -> DatasetRecordWithPrediction:
        assert (
            record.original is not None
        ), "stream must be given for docling prediction provider to work."
        res = self.doc_converter.convert(copy.deepcopy(record.original))
        pred_record = self.create_dataset_record_with_prediction(
            record,
            res.document,
            None,
        )
        pred_record.status = res.status

        return pred_record

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}
