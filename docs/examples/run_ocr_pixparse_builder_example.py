import logging
import os
from pathlib import Path

from docling_eval.dataset_builders.pixparse_builder import PixparseDatasetBuilder
from docling_eval.evaluators.ocr_evaluator import OCREvaluator
from docling_eval.prediction_providers.aws_prediction_provider import (
    AWSTextractPredictionProvider,
)
from docling_eval.prediction_providers.google_prediction_provider import (
    GoogleDocAIPredictionProvider,
)


def main():
    idir = Path("/docling-eval/pixparse-idl")
    odir = Path("/docling-eval/pixparse-idl/output")

    os.makedirs(odir, exist_ok=True)

    aws_provider = AWSTextractPredictionProvider(
        predictions_dir=odir / "aws", skip_api_if_prediction_is_present=True
    )
    aws_dataset = PixparseDatasetBuilder(
        prediction_provider=aws_provider,
        dataset_source=idir,
        target=odir / "aws",
    )
    aws_downloaded_path = aws_dataset.retrieve_input_dataset()
    logging.info(f"Dataset downloaded to {aws_downloaded_path} for AWS")

    aws_dataset.save_to_disk()

    azure_provider = AzureDocIntelligencePredictionProvider(
        predictions_dir=odir / "azure", skip_api_if_prediction_is_present=True
    )
    azure_dataset = PixparseDatasetBuilder(
        prediction_provider=azure_provider,
        dataset_source=idir,
        target=odir / "azure",
    )

    azure_downloaded_path = azure_dataset.retrieve_input_dataset()
    logging.info(f"Dataset downloaded to {azure_downloaded_path} for AZURE")
    azure_dataset.save_to_disk()

    provider_args = {
        "predictions_dir": odir / "google",
        "skip_api_if_prediction_is_present": True,
        "mime_type": "image/tiff",
    }
    google_provider = GoogleDocAIPredictionProvider(**provider_args)
    google_dataset = PixparseDatasetBuilder(
        prediction_provider=google_provider,
        dataset_source=idir,
        target=odir / "google",
    )

    google_downloaded_path = google_dataset.retrieve_input_dataset()
    logging.info(f"Dataset downloaded to {google_downloaded_path} for Google")

    google_dataset.save_to_disk()

    logging.info("Evaluate the OCR for the Pixparse dataset")
    evaluator = OCREvaluator()
    dataset_path = Path("/docling-eval/pixparse-idl/output")
    output_path = Path("/docling-eval/pixparse-idl")
    evaluation_results = evaluator(dataset_path, output_path)
    logging.info(f"Completed evaluation. Results saved to {output_path}")


if __name__ == "__main__":
    main()
