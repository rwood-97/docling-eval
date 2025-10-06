#!/usr/bin/env python3
"""
CVAT Evaluation Pipeline Utility

This utility provides a flexible pipeline for evaluating CVAT annotations by:
1. Converting CVAT XML annotations to DoclingDocument JSON format
2. Creating ground truth and prediction datasets
3. Running layout and document structure evaluations

The pipeline can be run in separate steps or end-to-end.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docling_eval.campaign_tools.combine_cvat_evaluations import (
    combine_cvat_evaluations,
)
from docling_eval.campaign_tools.evaluate_cvat_tables import evaluate_tables
from docling_eval.campaign_tools.merge_cvat_annotations import (
    create_merged_annotation_xml,
    extract_image_tags,
)
from docling_eval.cli.main import evaluate
from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.cvat_tools.folder_models import CVATFolderStructure, CVATPageInfo
from docling_eval.cvat_tools.folder_parser import (
    find_xml_files_by_pattern,
    parse_cvat_folder,
)
from docling_eval.cvat_tools.parser import MissingImageInCVATXML
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.prediction_providers.file_provider import FilePredictionProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)


class CVATEvaluationPipeline:
    """Pipeline for CVAT annotation evaluation."""

    def __init__(
        self,
        cvat_root: Path,
        output_dir: Path,
        *,
        strict: bool = False,
        set_a_pattern: str = "task_{xx}_set_A",
        set_b_pattern: str = "task_{xx}_set_B",
    ):
        """
        Initialize the pipeline.

        Args:
            cvat_root: Root directory of the ``cvat_dataset_preannotated`` export
            output_dir: Base directory for all pipeline outputs
            strict: If True, treat conversion failures as fatal (default: False)
            set_a_pattern: Glob-style pattern identifying ground-truth task XMLs
            set_b_pattern: Glob-style pattern identifying prediction task XMLs
        """
        self.cvat_root = Path(cvat_root)
        self.output_dir = Path(output_dir)
        self.strict = strict
        self.set_a_pattern = set_a_pattern
        self.set_b_pattern = set_b_pattern
        self._folder_cache: Dict[str, CVATFolderStructure] = {}

        # Create subdirectories
        self.gt_json_dir = self.output_dir / "ground_truth_json"
        self.pred_json_dir = self.output_dir / "predictions_json"
        self.gt_dataset_dir = self.output_dir / "gt_dataset"
        self.eval_dataset_dir = self.output_dir / "eval_dataset"
        self.evaluation_results_dir = self.output_dir / "evaluation_results"
        self._intermediate_dir = self.output_dir / "intermediate"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_folder_structure(self, xml_pattern: str) -> CVATFolderStructure:
        """Load and cache the CVAT folder structure for the provided pattern."""

        if xml_pattern in self._folder_cache:
            return self._folder_cache[xml_pattern]

        folder_structure = parse_cvat_folder(self.cvat_root, xml_pattern)
        self._folder_cache[xml_pattern] = folder_structure
        return folder_structure

    def _iterate_pages(
        self,
        folder_structure: CVATFolderStructure,
    ) -> Iterable[CVATPageInfo]:
        """Yield every page recorded in ``folder_structure``."""

        for document in folder_structure.documents.values():
            for page_info in document.pages:
                yield page_info

    def _convert_cvat_set_to_json(
        self,
        xml_pattern: str,
        output_json_dir: Path,
    ) -> List[Path]:
        """Convert all pages matching ``xml_pattern`` into Docling JSON files."""

        output_json_dir.mkdir(parents=True, exist_ok=True)

        folder_structure = self._load_folder_structure(xml_pattern)
        pages: List[CVATPageInfo] = sorted(
            self._iterate_pages(folder_structure),
            key=lambda page: (page.doc_hash, page.page_number, page.image_filename),
        )

        if not pages:
            raise ValueError(
                f"No annotated pages found using pattern '{xml_pattern}' in {self.cvat_root}"
            )

        if self.strict:
            _log.info(
                "Running in STRICT mode: conversion failures will abort the pipeline"
            )
        else:
            _log.info(
                "Running in NORMAL mode: conversion failures will be logged and skipped"
            )

        json_files: List[Path] = []
        failed_pages: List[str] = []

        for page in pages:
            image_path = page.image_path
            xml_path = page.xml_path

            _log.info(
                "Converting %s from task XML %s",
                image_path.name,
                xml_path.name,
            )

            try:
                document = convert_cvat_to_docling(xml_path, image_path)
            except MissingImageInCVATXML:
                message = f"Image {image_path.name} not present in {xml_path.name}."
                if self.strict:
                    raise ValueError(message) from None
                _log.error(message)
                failed_pages.append(image_path.name)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                if self.strict:
                    raise
                _log.error("Conversion failed for %s: %s", image_path.name, exc)
                failed_pages.append(image_path.name)
                continue

            if document is None:
                message = f"Conversion returned None for {image_path.name}"
                if self.strict:
                    raise ValueError(message)
                _log.warning(message)
                failed_pages.append(image_path.name)
                continue

            json_path = output_json_dir / f"{image_path.stem}.json"
            document.save_as_json(json_path)
            json_files.append(json_path)
            _log.info("✓ Saved Docling JSON to %s", json_path)

        if failed_pages:
            _log.warning(
                "Skipped %d page(s) due to conversion issues: %s",
                len(failed_pages),
                ", ".join(sorted(failed_pages)[:5])
                + ("..." if len(failed_pages) > 5 else ""),
            )

        if self.strict and failed_pages:
            raise ValueError("Strict mode enabled: conversion errors were encountered.")

        _log.info("Converted %d/%d page(s) to JSON format", len(json_files), len(pages))
        return json_files

    def _merge_task_xmls(self, xml_pattern: str, destination: Path) -> Path:
        """Merge all CVAT task XMLs for ``xml_pattern`` into a single file."""

        xml_files = find_xml_files_by_pattern(self.cvat_root, xml_pattern)
        if not xml_files:
            raise ValueError(
                f"No XML files matching pattern '{xml_pattern}' found in {self.cvat_root / 'cvat_tasks'}"
            )

        _log.info(
            "Merging %d CVAT task XMLs matching '%s'", len(xml_files), xml_pattern
        )

        image_elements = extract_image_tags(xml_files)
        if not image_elements:
            raise ValueError(
                f"No annotated images discovered while merging pattern '{xml_pattern}'"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        create_merged_annotation_xml(image_elements, destination)
        _log.info("✓ Generated merged annotations at %s", destination)
        return destination

    def create_ground_truth_dataset(self) -> None:
        """
        Step 1: Create ground truth dataset from CVAT folder exports.
        """
        _log.info("=== Creating Ground Truth Dataset ===")

        # Convert CVAT XML to JSON
        gt_json_files = self._convert_cvat_set_to_json(
            self.set_a_pattern, self.gt_json_dir
        )

        if not gt_json_files:
            raise ValueError("No ground truth JSON files were created")

        # Create ground truth dataset
        dataset_builder = FileDatasetBuilder(
            name="CVAT_Ground_Truth_Dataset",
            dataset_source=self.gt_json_dir,
            target=self.gt_dataset_dir,
            split="test",
            file_extensions=["json"],
        )

        dataset_builder.save_to_disk(chunk_size=50, do_visualization=True)
        _log.info(f"✓ Ground truth dataset created: {self.gt_dataset_dir}")

    def create_prediction_dataset(self) -> None:
        """
        Step 2: Create prediction dataset from CVAT folder exports using the ground truth dataset.
        """
        _log.info("=== Creating Prediction Dataset ===")

        if not self.gt_dataset_dir.exists():
            raise ValueError(
                f"Ground truth dataset not found at {self.gt_dataset_dir}. "
                "Please run create_ground_truth_dataset first."
            )

        # Convert prediction CVAT XML to JSON
        pred_json_files = self._convert_cvat_set_to_json(
            self.set_b_pattern, self.pred_json_dir
        )

        if not pred_json_files:
            raise ValueError("No prediction JSON files were created")

        # Create prediction dataset using FilePredictionProvider
        file_provider = FilePredictionProvider(
            prediction_format=PredictionFormats.JSON,
            source_path=self.pred_json_dir,
            do_visualization=True,
            ignore_missing_files=True,
            ignore_missing_predictions=True,
            use_ground_truth_page_images=True,
        )

        file_provider.create_prediction_dataset(
            name="CVAT_Prediction_Dataset",
            gt_dataset_dir=self.gt_dataset_dir,
            target_dataset_dir=self.eval_dataset_dir,
            split="test",
            chunk_size=50,
        )
        _log.info(f"✓ Prediction dataset created: {self.eval_dataset_dir}")

    def run_table_evaluation(
        self,
        out_json: Optional[Path] = None,
        containment_thresh: float = 0.50,
        table_pair_iou: float = 0.20,
        sem_match_iou: float = 0.30,
    ) -> Path:
        """
        Run the table structure/semantics evaluation using merged CVAT task XMLs.

        Writes a JSON file (default: evaluation_results/evaluation_CVAT_tables.json) and returns its path.
        """
        _log.info("=== Running Table Evaluation ===")

        if out_json is None:
            out_json = self.evaluation_results_dir / "evaluation_CVAT_tables.json"

        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)

        merged_dir = self._intermediate_dir / "merged_xml"
        gt_merged = merged_dir / "combined_set_A.xml"
        pred_merged = merged_dir / "combined_set_B.xml"

        gt_xml = self._merge_task_xmls(self.set_a_pattern, gt_merged)
        pred_xml = self._merge_task_xmls(self.set_b_pattern, pred_merged)

        result = evaluate_tables(
            set_a=gt_xml,
            set_b=pred_xml,
            containment_thresh=containment_thresh,
            table_pair_iou=table_pair_iou,
            sem_match_iou=sem_match_iou,
        )

        out_json.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log.info(f"✓ Tables evaluation written to: {out_json}")
        return out_json

    def run_evaluation(
        self, modalities: Optional[List[str]] = None, user_csv: Optional[Path] = None
    ) -> None:
        """
        Step 3: Run evaluation on the prediction dataset.

        Args:
            modalities: List of evaluation modalities.
                       Options: ['layout', 'document_structure']
                       Default: both modalities
            user_csv: Path to user CSV file for provenance/self-confidence (optional)
        """
        _log.info("=== Running Evaluation ===")

        if not self.eval_dataset_dir.exists():
            raise ValueError(
                f"Evaluation dataset not found at {self.eval_dataset_dir}. "
                "Please run create_prediction_dataset first."
            )

        if modalities is None:
            modalities = ["layout", "document_structure", "key_value"]

        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)

        for modality_name in modalities:
            _log.info(f"Running {modality_name} evaluation...")

            if modality_name == "layout":
                modality = EvaluationModality.LAYOUT
            elif modality_name == "document_structure":
                modality = EvaluationModality.DOCUMENT_STRUCTURE
            elif modality_name == "key_value":
                modality = EvaluationModality.KEY_VALUE
            else:
                _log.warning(f"Unknown modality: {modality_name}. Skipping.")
                continue
            # TODO: add key-value evaluation, see https://github.com/docling-project/docling-eval/pull/140

            try:
                evaluation_result = evaluate(
                    modality=modality,
                    benchmark=BenchMarkNames.CVAT,
                    idir=self.eval_dataset_dir,
                    odir=self.evaluation_results_dir,
                    split="test",
                )

                if evaluation_result:
                    _log.info(
                        f"\u2713 {modality_name} evaluation completed successfully"
                    )
                    _log.info(
                        f"Evaluated samples: {evaluation_result.evaluated_samples}"
                    )

                    if modality_name == "layout":
                        _log.info(f"Mean mAP: {evaluation_result.mAP:.4f}")
                    elif modality_name == "document_structure":
                        _log.info(
                            f"Mean edit distance: {evaluation_result.edit_distance_stats.mean:.4f}"
                        )
                else:
                    _log.error(f"\u2717 {modality_name} evaluation failed")

            except Exception as e:
                _log.error(f"\u2717 Error in {modality_name} evaluation: {e}")
                raise e

        # Combine results if user_csv is provided
        combined_out = self.output_dir / "combined_evaluation.xlsx"
        layout_json = self.evaluation_results_dir / "evaluation_CVAT_layout.json"
        docstruct_json = (
            self.evaluation_results_dir / "evaluation_CVAT_document_structure.json"
        )
        key_value_json = self.evaluation_results_dir / "evaluation_CVAT_key_value.json"
        tables_json = self.evaluation_results_dir / "evaluation_CVAT_tables.json"
        _log.info(f"Combining evaluation results to {combined_out}")
        combine_cvat_evaluations(
            layout_json=layout_json,
            docstruct_json=docstruct_json,
            keyvalue_json=key_value_json,
            user_csv=user_csv,
            tables_json=tables_json,
            out=combined_out,
        )

    def run_full_pipeline(
        self,
        modalities: Optional[List[str]] = None,
        user_csv: Optional[Path] = None,
    ) -> None:
        """
        Run the complete pipeline: create datasets, run evaluation, and combine results.

        Args:
            modalities: List of evaluation modalities to run
            user_csv: Path to user CSV file for provenance/self-confidence
        """
        _log.info("=== Running Full CVAT Evaluation Pipeline ===")

        try:
            self.create_ground_truth_dataset()
            self.create_prediction_dataset()
            self.run_table_evaluation()
            self.run_evaluation(modalities, user_csv)

            # Combine results if user_csv is provided
            combined_out = self.output_dir / "combined_evaluation.xlsx"
            layout_json = self.evaluation_results_dir / "evaluation_CVAT_layout.json"
            docstruct_json = (
                self.evaluation_results_dir / "evaluation_CVAT_document_structure.json"
            )
            key_value_json = (
                self.evaluation_results_dir / "evaluation_CVAT_key_value.json"
            )
            tables_json = self.evaluation_results_dir / "evaluation_CVAT_tables.json"

            _log.info(f"Combining evaluation results to {combined_out}")
            combine_cvat_evaluations(
                layout_json=layout_json,
                docstruct_json=docstruct_json,
                keyvalue_json=key_value_json,
                user_csv=user_csv,
                tables_json=tables_json,
                out=combined_out,
            )

            _log.info("=== Pipeline completed successfully! ===")
            _log.info(f"Results available in: {self.output_dir}")

        except Exception as e:
            _log.error(f"Pipeline failed with error: {e}")
            raise e


def main():
    """Command line interface for the CVAT evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="CVAT Evaluation Pipeline - Convert CVAT annotations and run evaluations"
    )

    parser.add_argument(
        "cvat_root",
        type=Path,
        help="Path to the cvat_dataset_preannotated root directory",
    )

    parser.add_argument(
        "output_dir", type=Path, help="Output directory for pipeline results"
    )

    parser.add_argument(
        "--set-a-pattern",
        type=str,
        default="task_{xx}_set_A",
        help="Pattern used to locate ground-truth task XMLs (default: task_{xx}_set_A)",
    )

    parser.add_argument(
        "--set-b-pattern",
        type=str,
        default="task_{xx}_set_B",
        help="Pattern used to locate prediction task XMLs (default: task_{xx}_set_B)",
    )

    parser.add_argument(
        "--user-csv",
        type=Path,
        default=None,
        help="Path to user CSV file for provenance/self-confidence (optional, used for combining evaluation results)",
    )

    parser.add_argument(
        "--step",
        choices=["gt", "pred", "tables", "eval", "full"],
        default="full",
        help="Pipeline step to run: gt (ground truth), pred (predictions), tables (table eval only), eval, or full.",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["layout", "document_structure", "key_value"],
        default=["layout", "document_structure", "key_value"],
        help="Evaluation modalities to run",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: abort if any conversion fails (default: log and continue)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.cvat_root.exists():
        _log.error(f"CVAT root directory does not exist: {args.cvat_root}")
        sys.exit(1)

    if not args.cvat_root.is_dir():
        _log.error(f"CVAT root path is not a directory: {args.cvat_root}")
        sys.exit(1)

    overview_path = args.cvat_root / "cvat_overview.json"
    if not overview_path.exists():
        _log.error(
            "cvat_overview.json not found in %s. Please point to a cvat_dataset_preannotated root.",
            args.cvat_root,
        )
        sys.exit(1)

    pipeline = CVATEvaluationPipeline(
        cvat_root=args.cvat_root,
        output_dir=args.output_dir,
        strict=args.strict,
        set_a_pattern=args.set_a_pattern,
        set_b_pattern=args.set_b_pattern,
    )

    if args.step == "gt":
        pipeline.create_ground_truth_dataset()
    elif args.step == "pred":
        pipeline.create_prediction_dataset()
    elif args.step == "tables":
        pipeline.run_table_evaluation()
    elif args.step == "eval":
        pipeline.run_evaluation(args.modalities, user_csv=args.user_csv)
    elif args.step == "full":
        pipeline.run_full_pipeline(args.modalities, user_csv=args.user_csv)


if __name__ == "__main__":
    main()
