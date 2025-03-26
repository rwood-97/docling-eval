from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional

from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument, DocTagsPage
from PIL import Image

from docling_eval.datamodels.constants import PredictionFormats
from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)


class FilePredictionProvider(BasePredictionProvider):
    def __init__(
        self,
        prediction_format: PredictionFormats,
        source_path: Path,
        do_visualization: bool = False,
        ignore_missing_files: Optional[bool] = False,
    ):
        super().__init__(do_visualization=do_visualization)
        self._supported_prediction_formats = [
            PredictionFormats.DOCTAGS,
            PredictionFormats.MARKDOWN,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]

        # Read the input
        self._prediction_format = prediction_format
        self._prediction_source_path = source_path
        self._ignore_missing_files = ignore_missing_files

        # Validate the prediction format
        if self._prediction_format not in self._supported_prediction_formats:
            msg = f"Unsupported file prediction format: {self._prediction_format}."
            msg += f" The prediction format must be one of {self._supported_prediction_formats}"
            raise RuntimeError(msg)

        # Validate if the source_path exists
        if not self._prediction_source_path.is_dir():
            raise RuntimeError(f"Missing source path: {self._prediction_source_path}")

    @abstractmethod
    def info(self) -> Dict:
        return {
            "supported_prediction_formats": self._supported_prediction_formats,
            "prediction_format": self._prediction_format,
            "source_path": self._prediction_source_path,
        }

    @abstractmethod
    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        doc_id = record.doc_id
        raw = None
        if self._prediction_format == PredictionFormats.DOCTAGS:
            pred_doc = self._load_doctags_doc(doc_id)
        elif self._prediction_format == PredictionFormats.MARKDOWN:
            pred_doc = None
            raw = self._load_md_raw(doc_id)
        elif self._prediction_format == PredictionFormats.JSON:
            pred_doc = self._load_json_doc(doc_id)
        elif self._prediction_format == PredictionFormats.YAML:
            pred_doc = self._load_yaml_doc(doc_id)

        if pred_doc is None:
            status = ConversionStatus.FAILURE
        else:
            status = ConversionStatus.SUCCESS

        pred_record = self.create_dataset_record_with_prediction(
            record,
            pred_doc,
            raw,
        )
        pred_record.status = status
        return pred_record

    @property
    @abstractmethod
    def prediction_format(self) -> PredictionFormats:
        r""" """
        return self._prediction_format

    def _load_doctags_doc(self, doc_id: str) -> Optional[DoclingDocument]:
        r"""Load doctags file into DoclingDocument"""
        # Read the doctags file
        doctags_fn = self._prediction_source_path / f"{doc_id}.dt"
        if self._ignore_missing_files and not doctags_fn.is_file():
            return None

        with open(doctags_fn, "r") as fd:
            doctags = fd.read()

        # Check if an optional page image is present
        page_image_fn = self._prediction_source_path / f"{doc_id}.png"
        page_image = None
        if page_image_fn.is_file():
            page_image = Image.open(page_image_fn)

        # Build DoclingDocument
        doctags_page = DocTagsPage(tokens=doctags, image=page_image)
        doctags_doc = DocTagsDocument(pages=[doctags_page])
        doc = DoclingDocument(name=doc_id)
        doc.load_from_doctags(doctags_doc)

        return doc

    def _load_json_doc(self, doc_id: str) -> Optional[DoclingDocument]:
        r"""Load DoclingDocument from json"""
        json_fn = self._prediction_source_path / f"{doc_id}.json"
        if self._ignore_missing_files and not json_fn.is_file():
            return None
        doc: DoclingDocument = DoclingDocument.load_from_json(json_fn)
        return doc

    def _load_yaml_doc(self, doc_id: str) -> Optional[DoclingDocument]:
        r"""Load DoclingDocument from yaml"""
        yaml_fn = self._prediction_source_path / f"{doc_id}.yaml"
        if not yaml_fn.is_file():
            # Try alternative yaml extension
            yaml_fn = self._prediction_source_path / f"{doc_id}.yml"
        if self._ignore_missing_files and not yaml_fn.is_file():
            return None

        doc: DoclingDocument = DoclingDocument.load_from_yaml(yaml_fn)
        return doc

    def _load_md_raw(self, doc_id: str) -> Optional[str]:
        r"""Load the markdown content"""
        md_fn = self._prediction_source_path / f"{doc_id}.md"
        if self._ignore_missing_files and not md_fn.is_file():
            return None

        with open(md_fn, "r") as fd:
            md = fd.read()
        return md
