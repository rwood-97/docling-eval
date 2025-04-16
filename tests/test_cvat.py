import shutil
from pathlib import Path

from docling_eval.cli.main import (
    PredictionProviderType,
    evaluate,
    get_prediction_provider,
    visualize,
)
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.cvat_dataset_builder import CvatDatasetBuilder
from docling_eval.dataset_builders.cvat_preannotation_builder import (
    CvatPreannotationBuilder,
)
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder


def test_run_cvat_on_gt():
    gt_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-GT/")
    cvat_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-CVAT/")

    # Stage 1: Create and pre-annotate dataset
    dataset_layout = DPBenchDatasetBuilder(
        target=gt_path,
        begin_index=15,
        end_index=20,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    builder = CvatPreannotationBuilder(
        source_dir=gt_path / "test", target_dir=cvat_path, bucket_size=20
    )
    builder.prepare_for_annotation()

    ## Stage 2: Re-build dataset
    shutil.copy(
        "./tests/data/annotations_cvat.zip",
        str(cvat_path / "cvat_annotations" / "zips"),
    )

    # Create dataset from CVAT annotations
    dataset_builder = CvatDatasetBuilder(
        name="MyCVATAnnotations",
        cvat_source_dir=cvat_path,
        target=cvat_path,
        split="test",
    )
    dataset_builder.retrieve_input_dataset()
    dataset_builder.save_to_disk()

    # Stage 3: Create predictions and compare to CVAT GT dataset
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)
    eval_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-PRED")

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=cvat_path,
        target_dataset_dir=eval_path / "eval_dataset",
    )

    ## Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=eval_path / "eval_dataset",
        odir=eval_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=eval_path / "eval_dataset",
        odir=eval_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )
