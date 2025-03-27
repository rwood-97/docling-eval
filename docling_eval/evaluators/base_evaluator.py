import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from docling_eval.datamodels.constants import PredictionFormats

_log = logging.getLogger(__name__)


class DatasetEvaluation(BaseModel):
    pass


class BaseEvaluator:
    r"""
    Base class for all evaluators
    """

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
    ):
        r"""
        Parameters
        ----------
        intermediate_evaluations_path: When True the evalution per example will be saved in a file
        """
        self._intermediate_evaluations_path = intermediate_evaluations_path
        self._prediction_sources = prediction_sources

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
        # Remove the ext_predictions when all evaluators have been migrated to the new design
        ext_predictions: Optional[
            Dict[str, Any]
        ] = None,  # Optionally provided external predictions
    ) -> DatasetEvaluation:
        r"""
        Perform the evaluation
        """
        return None  # type: ignore

    def supported_prediction_formats(self) -> List[PredictionFormats]:
        r"""
        Return the supported formats for predictions
        """
        return [PredictionFormats.DOCLING_DOCUMENT]
