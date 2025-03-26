import copy
from typing import Dict

from docling.datamodel.pipeline_options import TableFormerMode
from docling_core.types.io import DocumentStream

from docling_eval.converters.models.tableformer.tf_model_prediction import (
    TableFormerUpdater,
)
from docling_eval.datamodels.constants import PredictionFormats
from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import docling_models_version


class TableFormerPredictionProvider(BasePredictionProvider):
    def __init__(self, do_visualization: bool = False):
        super().__init__(do_visualization=do_visualization)
        self.tf_updater = TableFormerUpdater(TableFormerMode.ACCURATE)

    @property
    def prediction_format(self) -> PredictionFormats:
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        r""" """
        assert (
            record.ground_truth_doc is not None
        ), "true_doc must be given for TableFormer prediction provider to work."

        if record.mime_type == "application/pdf":
            assert isinstance(record.original, DocumentStream)

            updated, pred_doc = self.tf_updater.replace_tabledata(
                copy.deepcopy(record.original.stream), record.ground_truth_doc
            )
        elif record.mime_type == "image/png":
            updated, pred_doc = self.tf_updater.replace_tabledata_with_page_tokens(
                record.ground_truth_doc,
                record.ground_truth_page_images,
            )
        else:
            raise RuntimeError(
                "TableFormerPredictionProvider is missing data to predict on."
            )
        pred_record = self.create_dataset_record_with_prediction(
            record,
            pred_doc,
            None,
        )
        return pred_record

    def info(self) -> Dict:
        return {"asset": "TableFormer", "version": docling_models_version()}
