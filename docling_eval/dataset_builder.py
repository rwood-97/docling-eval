import os
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Union

from docling.utils.utils import chunkify
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from docling_eval.benchmarks.utils import write_datasets_info
from docling_eval.dataset_record import DatasetRecord
from docling_eval.docling.utils import save_shard_to_disk
from docling_eval.prediction_provider import BasePredictionProvider


class HFSource(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)


class S3Source(BaseModel):
    # TBD
    pass


class BaseEvaluationDatasetBuilder:
    def __init__(
        self,
        name: str,
        dataset_source: Union[HFSource, S3Source, Path],
        prediction_provider: BasePredictionProvider,
        target: Optional[Path] = None,
    ):
        self.name = name
        self.target = target
        self.prediction_provider = prediction_provider
        self.dataset_source = dataset_source

        self.dataset_local_path = None  # TBD

        self.retrieved = False

    def retrieve_input_dataset(self) -> Path:
        if isinstance(self.dataset_source, HFSource):
            if not self.dataset_local_path:
                path = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    # local_dir=self.target,
                )
                path = Path(path)
                self.dataset_local_path = path
            else:
                path = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    local_dir=self.dataset_local_path,
                )
        elif isinstance(self.dataset_source, Path):
            path = self.dataset_source
        else:
            raise RuntimeError(
                f"Unknown dataset_source type {type(self.dataset_source)}"
            )

        self.retrieved = True

        return path

    @abstractmethod
    def iterate(self) -> Iterable[DatasetRecord]:
        pass

    def update_prediction(self, record: DatasetRecord):
        # This might need customization depending on the input the dataset has.
        # The default implementation assumes that there is an original file in binary format which is accepted.
        pred_doc = self.prediction_provider.predict(record.original)
        record.predicted_doc = pred_doc

        record.validate_model()

    def save_to_disk(self):
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / "test"
        os.makedirs(test_dir, exist_ok=True)

        count = 0
        for record_chunk in chunkify(self.iterate(), 80):
            record_chunk = [r.as_record_dict() for r in record_chunk]
            save_shard_to_disk(items=record_chunk, dataset_path=test_dir)
            count += len(record_chunk)

        write_datasets_info(
            name=self.name,
            output_dir=self.target,
            num_train_rows=0,
            num_test_rows=count,
        )
