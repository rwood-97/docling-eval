import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Union

from docling.utils.utils import chunkify
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.utils.utils import save_shard_to_disk, write_datasets_info


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
        target: Path,
        split: str = "test",
    ):
        self.name = name
        self.target: Path = target

        self.dataset_source = dataset_source
        self.dataset_local_path: Optional[Path] = None  # TBD
        self.split = split
        self.retrieved = False

    def retrieve_input_dataset(self) -> Path:
        if isinstance(self.dataset_source, HFSource):
            if not self.dataset_local_path:
                path_str = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    # local_dir=self.target,
                )
                path: Path = Path(path_str)
                self.dataset_local_path = path
            else:
                path_str = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    local_dir=self.dataset_local_path,
                )
                path = Path(path_str)
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

    def save_to_disk(self, chunk_size: int = 80, max_num_chunks: int = sys.maxsize):
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / self.split
        os.makedirs(test_dir, exist_ok=True)

        count = 0
        chunk_count = 0
        for record_chunk in chunkify(self.iterate(), chunk_size):
            record_chunk = [r.as_record_dict() for r in record_chunk]
            save_shard_to_disk(
                items=record_chunk, dataset_path=test_dir, shard_id=chunk_count
            )
            count += len(record_chunk)
            chunk_count += 1

            if chunk_count >= max_num_chunks:
                break

        write_datasets_info(
            name=self.name,
            output_dir=self.target,
            num_train_rows=0,
            num_test_rows=count,
        )
