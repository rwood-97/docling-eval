from pathlib import Path
from typing import List, Optional

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.cli.main import evaluate
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.fintabnet_builder import FintabnetTableStructureDatasetBuilder
from docling_eval_next.prediction_providers.prediction_provider import (
    AzureDocIntelligencePredictionProvider,
)


def main():
    target_path = Path("./scratch/fintabnet-builer-test/")
    provider = AzureDocIntelligencePredictionProvider()

    dataset = FintabnetTableStructureDatasetBuilder(
        prediction_provider=provider,
        target=target_path,
    )

    downloaded_path = dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    print(f"Dataset downloaded to {downloaded_path}")

    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    # evaluate(
    #     modality=EvaluationModality.LAYOUT,
    #     benchmark=BenchMarkNames.DPBENCH,
    #     idir=target_path,
    #     odir=target_path / "layout",
    # )


if __name__ == "__main__":
    main()
