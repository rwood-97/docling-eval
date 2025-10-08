from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from docling_eval.campaign_tools.cvat_evaluation_pipeline import CVATEvaluationPipeline

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubmissionSubsetJob:
    """Container describing the artefacts needed to evaluate one submission subset."""

    submission_name: str
    subset_name: str
    tasks_root: Path
    base_cvat_root: Path
    output_dir: Path


def discover_jobs(
    deliveries_root: Path,
    datasets_root: Path,
    output_root: Path,
) -> List[SubmissionSubsetJob]:
    """Enumerate all submission subset combinations that can be evaluated."""
    jobs: List[SubmissionSubsetJob] = []

    if not deliveries_root.exists():
        raise FileNotFoundError(f"Deliveries root does not exist: {deliveries_root}")

    for submission_dir in sorted(deliveries_root.glob("submission-*")):
        if not submission_dir.is_dir():
            continue

        submission_name = submission_dir.name
        delivery_dir = submission_dir / "delivery"
        if not delivery_dir.is_dir():
            _LOGGER.warning("Skipping %s: missing delivery directory", submission_name)
            continue

        for subset_dir in sorted(delivery_dir.iterdir()):
            if not subset_dir.is_dir():
                continue

            subset_name = subset_dir.name
            tasks_root = subset_dir / "cvat_dataset_preannotated"
            if not tasks_root.exists():
                _LOGGER.warning(
                    "Skipping %s/%s: tasks root %s missing",
                    submission_name,
                    subset_name,
                    tasks_root,
                )
                continue

            base_cvat_root = datasets_root / subset_name / "cvat_dataset_preannotated"
            if not base_cvat_root.exists():
                _LOGGER.warning(
                    "Skipping %s/%s: base dataset root %s missing",
                    submission_name,
                    subset_name,
                    base_cvat_root,
                )
                continue

            output_dir = output_root / submission_name / subset_name
            jobs.append(
                SubmissionSubsetJob(
                    submission_name=submission_name,
                    subset_name=subset_name,
                    tasks_root=tasks_root,
                    base_cvat_root=base_cvat_root,
                    output_dir=output_dir,
                )
            )

    return jobs


def run_jobs(
    jobs: Sequence[SubmissionSubsetJob],
    *,
    modalities: Optional[Sequence[str]] = None,
    strict: bool = False,
    dry_run: bool = False,
    user_csv: Optional[Path] = None,
    force: bool = False,
    merge_only: bool = False,
) -> None:
    """Execute the CVAT evaluation pipeline for each prepared job."""
    if not jobs:
        _LOGGER.info("No jobs discovered; nothing to do.")
        return

    for job in jobs:
        _LOGGER.info(
            "Processing submission=%s subset=%s",
            job.submission_name,
            job.subset_name,
        )

        merged_dir = job.output_dir / "merged_xml"
        merged_gt = merged_dir / "combined_set_A.xml"
        merged_pred = merged_dir / "combined_set_B.xml"

        if merge_only:
            if not force and merged_gt.exists() and merged_pred.exists():
                _LOGGER.info(
                    "Skipping %s/%s: merged XML already present at %s",
                    job.submission_name,
                    job.subset_name,
                    merged_dir,
                )
                continue
        else:
            if job.output_dir.exists() and not force:
                _LOGGER.info(
                    "Skipping %s/%s: output directory already exists at %s (use --force to re-run)",
                    job.submission_name,
                    job.subset_name,
                    job.output_dir,
                )
                continue

        if dry_run:
            verb = "merge annotations for" if merge_only else "evaluate"
            _LOGGER.info(
                "Dry-run: would %s %s/%s with base=%s tasks=%s output=%s",
                verb,
                job.submission_name,
                job.subset_name,
                job.base_cvat_root,
                job.tasks_root,
                job.output_dir,
            )
            continue

        pipeline = CVATEvaluationPipeline(
            cvat_root=job.base_cvat_root,
            output_dir=job.output_dir,
            strict=strict,
            tasks_root=job.tasks_root,
        )

        if merge_only:
            pipeline.merge_annotation_xmls(destination_dir=merged_dir)
            continue

        pipeline.create_ground_truth_dataset()
        pipeline.create_prediction_dataset()
        pipeline.run_table_evaluation()
        pipeline.run_evaluation(
            modalities=list(modalities) if modalities else None, user_csv=user_csv
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CVAT evaluation pipeline across all submission deliveries."
        )
    )
    parser.add_argument(
        "--deliveries-root",
        type=Path,
        help="Root directory containing submission-*/delivery/* structures.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        help="Root directory containing the canonical base dataset subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory where evaluation artefacts will be written.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        choices=["layout", "document_structure", "key_value"],
        help="Optional list of evaluation modalities to run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode to stop on conversion errors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without running the pipeline.",
    )
    parser.add_argument(
        "--user-csv",
        type=Path,
        help="Optional CSV to merge into combined evaluation output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even when the output directory already exists.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only create combined set A/B XMLs for each submission subset.",
    )

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)

    jobs = discover_jobs(args.deliveries_root, args.datasets_root, args.output_root)
    run_jobs(
        jobs,
        modalities=args.modalities,
        strict=args.strict,
        dry_run=args.dry_run,
        user_csv=args.user_csv,
        force=args.force,
        merge_only=args.merge_only,
    )


if __name__ == "__main__":
    main()
