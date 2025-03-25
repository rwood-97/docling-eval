import copy
import json
import os
import sys
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from docling.datamodel.pipeline_options import TableFormerMode
from docling.document_converter import DocumentConverter
from docling.utils.utils import chunkify
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    DocTagsPage,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.benchmarks.constants import PredictionFormats
from docling_eval.benchmarks.utils import (
    docling_models_version,
    docling_version,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.converters.models.tableformer.tf_model_prediction import (
    PageToken,
    TableFormerUpdater,
)
from docling_eval_next.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)


class BasePredictionProvider:
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        self.provider_args = kwargs

    @abstractmethod
    def predict(self, record: DatasetRecord) -> Tuple[DoclingDocument, Optional[str]]:
        return DoclingDocument(name="dummy"), None

    @abstractmethod
    def info(self) -> Dict:
        return {}

    @property
    @abstractmethod
    def prediction_format(self) -> PredictionFormats:
        pass

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
        pred_doc, orig_pred = self.predict(record)

        pred_record = DatasetRecordWithPrediction.model_validate(
            {
                **record.as_record_dict(),
                "predicted_doc": pred_doc,
                "original_prediction": orig_pred,
                "prediction_format": self.prediction_format,
            }
        )

        pred_record.validate_images()  # type: ignore

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


class DoclingPredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        if kwargs is not None:
            if "format_options" in kwargs:
                self.doc_converter = DocumentConverter(
                    format_options=kwargs["format_options"]
                )
            else:
                self.doc_converter = DocumentConverter()

    @property
    def prediction_format(self) -> PredictionFormats:
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(
        self,
        record: DatasetRecord,
    ) -> Tuple[DoclingDocument, Optional[str]]:
        assert (
            record.original is not None
        ), "stream must be given for docling prediction provider to work."

        return self.doc_converter.convert(copy.deepcopy(record.original)).document, None

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}


class TableFormerPredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        self.tf_updater = TableFormerUpdater(TableFormerMode.ACCURATE)

    @property
    def prediction_format(self) -> PredictionFormats:
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(
        self, record: DatasetRecord, page_tokens: Optional[List[PageToken]] = None
    ) -> Tuple[DoclingDocument, Optional[str]]:

        assert (
            record.ground_truth_doc is not None
        ), "true_doc must be given for TableFormer prediction provider to work."

        assert record.original is None or isinstance(record.original, DocumentStream)

        if record.original is not None and page_tokens is None:
            updated, pred_doc = self.tf_updater.replace_tabledata(
                copy.deepcopy(record.original.stream), record.ground_truth_doc
            )
        elif page_tokens is not None:
            updated, pred_doc = self.tf_updater.replace_tabledata_with_page_tokens(
                page_tokens, record.ground_truth_doc, []
            )  # FIXME: Must not expect page images.
        else:
            raise RuntimeError(
                "TableFormerPredictionProvider.predict must be called with a stream or page_tokens."
            )
        return pred_doc, None

    def info(self) -> Dict:
        return {"asset": "TableFormer", "version": docling_models_version()}


class FilePredictionProvider(BasePredictionProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._supported_prediction_formats = [
            PredictionFormats.DOCTAGS,
            PredictionFormats.MARKDOWN,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]

        # Read the input
        self._prediction_format: PredictionFormats = PredictionFormats.DOCTAGS
        self._prediction_source_path: Path = Path(".")
        self._raise_on_missing_file = (
            False  # Raise exception when an expected file is missing
        )
        if kwargs is not None:
            if "prediction_format" in kwargs and isinstance(
                kwargs["prediction_format"], PredictionFormats
            ):
                self._prediction_format = kwargs["prediction_format"]
            if "source_path" in kwargs:
                if isinstance(kwargs["source_path"], Path):
                    self._prediction_source_path = kwargs["source_path"]
                else:
                    self._prediction_source_path = Path(kwargs["source_path"])
            if "raise_on_missing_file" in kwargs:
                self._raise_on_missing_file = kwargs["raise_on_missing_file"]

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
    def predict(self, record: DatasetRecord) -> Tuple[DoclingDocument, Optional[str]]:
        doc_id = record.doc_id
        raw = None
        if self._prediction_format == PredictionFormats.DOCTAGS:
            doc = self._load_doctags_doc(doc_id)
        elif self._prediction_format == PredictionFormats.MARKDOWN:
            # TODO: Replace the return type with something that has the DoclingDocument as optional
            doc = DoclingDocument(name=doc_id)  # Temp solution to pass MyPy

            raw = self._load_md_raw(doc_id)
        elif self._prediction_format == PredictionFormats.JSON:
            doc = self._load_json_doc(doc_id)
        elif self._prediction_format == PredictionFormats.YAML:
            doc = self._load_yaml_doc(doc_id)

        return doc, raw

    @property
    @abstractmethod
    def prediction_format(self) -> PredictionFormats:
        r""" """
        return self._prediction_format

    def _load_doctags_doc(self, doc_id: str) -> DoclingDocument:
        r"""Load doctags file into DoclingDocument"""
        # Read the doctags file
        doctags_fn = self._prediction_source_path / f"{doc_id}.dt"
        if self._raise_on_missing_file and not doctags_fn.is_file():
            raise RuntimeError(f"Missing prediction doctags: {doctags_fn}")
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

    def _load_json_doc(self, doc_id: str) -> DoclingDocument:
        r"""Load DoclingDocument from json"""
        json_fn = self._prediction_source_path / f"{doc_id}.json"
        if self._raise_on_missing_file and not json_fn.is_file():
            raise RuntimeError(f"Missing prediction json: {json_fn}")
        doc: DoclingDocument = DoclingDocument.load_from_json(json_fn)
        return doc

    def _load_yaml_doc(self, doc_id: str) -> DoclingDocument:
        r"""Load DoclingDocument from yaml"""
        yaml_fn = self._prediction_source_path / f"{doc_id}.yaml"
        if not yaml_fn.is_file():
            # Try alternative yaml extension
            yaml_fn = self._prediction_source_path / f"{doc_id}.yml"
        if self._raise_on_missing_file and not yaml_fn.is_file():
            raise RuntimeError(f"Missing prediction yaml: {yaml_fn}")

        doc: DoclingDocument = DoclingDocument.load_from_yaml(yaml_fn)
        return doc

    def _load_md_raw(self, doc_id: str) -> str:
        r"""Load the markdown content"""
        md_fn = self._prediction_source_path / f"{doc_id}.md"
        if self._raise_on_missing_file and not md_fn.is_file():
            raise RuntimeError(f"Missing prediction markdown: {md_fn}")
        with open(md_fn, "r") as fd:
            md = fd.read()
        return md
