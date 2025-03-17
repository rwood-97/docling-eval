from abc import abstractmethod
from typing import Dict

from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.benchmarks.utils import docling_version


class BasePredictionProvider:
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        self.provider_args = kwargs

    @abstractmethod
    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        return DoclingDocument(name="dummy")

    @abstractmethod
    def info(self) -> Dict:
        return {}
