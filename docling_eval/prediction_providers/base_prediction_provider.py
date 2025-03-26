import copy
import os
import sys
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

from datasets import load_dataset
from docling.utils.utils import chunkify
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from tqdm import tqdm

from docling_eval.datamodels.constants import PredictionFormats
from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.utils.utils import save_shard_to_disk, write_datasets_info
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class BasePredictionProvider:
    def __init__(self, do_visualization: bool = False):
        self.do_visualization = do_visualization

    @abstractmethod
    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        pred_record = self.create_dataset_record_with_prediction(
            record,
            DoclingDocument(name="dummy"),
            None,
        )
        return pred_record

    @abstractmethod
    def info(self) -> Dict:
        return {}

    def visualize_results(
        self, prediction_record: DatasetRecordWithPrediction, target_dataset_dir: Path
    ):
        if prediction_record.predicted_doc is not None:
            save_comparison_html_with_clusters(
                filename=target_dataset_dir
                / "visualizations"
                / f"{prediction_record.doc_id}.html",
                true_doc=prediction_record.ground_truth_doc,
                pred_doc=prediction_record.predicted_doc,
                page_image=prediction_record.ground_truth_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
                draw_reading_order=True,
            )

    @property
    @abstractmethod
    def prediction_format(self) -> PredictionFormats:
        pass

    def create_dataset_record_with_prediction(
        self,
        record: DatasetRecord,
        predicted_doc: Optional[DoclingDocument] = None,
        original_prediction: Optional[str] = None,
    ):
        pred_record = DatasetRecordWithPrediction.model_validate(
            {
                **record.as_record_dict(),
                "predicted_doc": predicted_doc,
                "original_prediction": original_prediction,
                "prediction_format": self.prediction_format,
            }
        )
        pred_record.validate_images()  # type: ignore
        return pred_record

    def add_prediction(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        # This might need customization depending on the input the dataset has.
        # The default implementation assumes that there is an original file in binary format which is accepted.
        input_data = copy.deepcopy(record.original)

        if not isinstance(input_data, DocumentStream):
            if isinstance(input_data, Path):
                input_data = DocumentStream(
                    name=input_data.name, stream=BytesIO(input_data.open("rb").read())
                )

        record.original = input_data
        pred_record = self.predict(record)

        return pred_record

    def create_prediction_dataset(
        self,
        name: str,
        gt_dataset_dir: Path,
        target_dataset_dir: Path,
        split: str = "test",
    ):
        parquet_files = str(gt_dataset_dir / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        # _log.info(f"oveview of dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        def _iterate_predictions():
            for i, data in tqdm(
                enumerate(ds_selection),
                desc="Creating predictions",
                ncols=120,
                total=len(ds_selection),
            ):
                record = DatasetRecord.model_validate(data)
                pred_record = self.add_prediction(record)

                yield pred_record

        test_dir = target_dataset_dir / "test"
        os.makedirs(test_dir, exist_ok=True)

        chunk_size = 80
        max_num_chunks = sys.maxsize

        count = 0
        chunk_count = 0
        for record_chunk in chunkify(_iterate_predictions(), chunk_size):
            if self.do_visualization:
                for r in record_chunk:
                    self.visualize_results(r, target_dataset_dir)

            record_chunk = [r.as_record_dict() for r in record_chunk]

            save_shard_to_disk(
                items=record_chunk, dataset_path=test_dir, shard_id=chunk_count
            )
            count += len(record_chunk)
            chunk_count += 1

            if chunk_count >= max_num_chunks:
                break

        write_datasets_info(
            name=name,
            output_dir=target_dataset_dir,
            num_train_rows=0,
            num_test_rows=count,
        )
