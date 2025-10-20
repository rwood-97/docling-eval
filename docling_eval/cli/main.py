import glob
import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path

# --- DoclingLayoutOptionsManager definition moved here ---
from typing import Annotated, Dict, List, Optional, Tuple

import typer
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PaginatedPipelineOptions,
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_model_specs import (
    GRANITEDOCLING_MLX,
    GRANITEDOCLING_TRANSFORMERS,
)
from docling.datamodel.vlm_model_specs import (
    SMOLDOCLING_MLX as smoldocling_vlm_mlx_conversion_options,
)
from docling.datamodel.vlm_model_specs import (
    SMOLDOCLING_TRANSFORMERS as smoldocling_vlm_conversion_options,
)
from docling.document_converter import FormatOption, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling.pipeline.vlm_pipeline import VlmPipeline
from PyPDF2 import PdfReader, PdfWriter
from tabulate import tabulate  # type: ignore

from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.dataset_builders.cvat_dataset_builder import CvatDatasetBuilder
from docling_eval.dataset_builders.cvat_preannotation_builder import (
    CvatPreannotationBuilder,
)
from docling_eval.dataset_builders.doclaynet_v1_builder import DocLayNetV1DatasetBuilder
from docling_eval.dataset_builders.doclaynet_v2_builder import DocLayNetV2DatasetBuilder
from docling_eval.dataset_builders.doclingdpbench_builder import (
    DoclingDPBenchDatasetBuilder,
)
from docling_eval.dataset_builders.docvqa_builder import DocVQADatasetBuilder
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval.dataset_builders.custom_omnidocbench_builder import (
    CustomOmniDocBenchDatasetBuilder,
)
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
    PubTables1MDatasetBuilder,
    PubTabNetDatasetBuilder,
)
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.evaluators.base_evaluator import DatasetEvaluationType
from docling_eval.evaluators.bbox_text_evaluator import BboxTextEvaluator
from docling_eval.evaluators.doc_structure_evaluator import DocStructureEvaluator
from docling_eval.evaluators.keyvalue_evaluator import (
    DatasetKeyValueEvaluation,
    KeyValueEvaluator,
)
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LabelFilteringStrategy,
    LayoutEvaluator,
    MissingPredictionStrategy,
)
from docling_eval.evaluators.markdown_text_evaluator import (
    DatasetMarkdownEvaluation,
    MarkdownTextEvaluator,
)
from docling_eval.evaluators.ocr.evaluation_models import TextCellUnit
from docling_eval.evaluators.ocr_evaluator import (
    OcrDatasetEvaluationResult,
    OCREvaluator,
    OCRVisualizer,
)
from docling_eval.evaluators.readingorder_evaluator import (
    DatasetReadingOrderEvaluation,
    ReadingOrderEvaluator,
    ReadingOrderVisualizer,
)
from docling_eval.evaluators.stats import DatasetStatistics
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)
from docling_eval.evaluators.timings_evaluator import (
    DatasetTimingsEvaluation,
    TimingsEvaluator,
)
from docling_eval.prediction_providers.aws_prediction_provider import (
    AWSTextractPredictionProvider,
)
from docling_eval.prediction_providers.azure_prediction_provider import (
    AzureDocIntelligencePredictionProvider,
)
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
from docling_eval.prediction_providers.file_provider import FilePredictionProvider
from docling_eval.prediction_providers.google_prediction_provider import (
    GoogleDocAIPredictionProvider,
)
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
)


class DoclingLayoutOptionsManager:
    layout_model_configs = {
        "docling_layout_v2": DOCLING_LAYOUT_V2,
        "docling_layout_heron": DOCLING_LAYOUT_HERON,
        "docling_layout_heron_101": DOCLING_LAYOUT_HERON_101,
        "docling_layout_egret_medium": DOCLING_LAYOUT_EGRET_MEDIUM,
        "docling_layout_egret_large": DOCLING_LAYOUT_EGRET_LARGE,
        "docling_layout_egret_xlarge": DOCLING_LAYOUT_EGRET_XLARGE,
    }

    @staticmethod
    def get_layout_model_config(model_spec: str) -> LayoutModelConfig:
        return DoclingLayoutOptionsManager.layout_model_configs[model_spec]

    @staticmethod
    def get_layout_model_config_names() -> List[str]:
        return list(DoclingLayoutOptionsManager.layout_model_configs.keys())


# Configure logging
logging_level = logging.WARNING
# logging_level = logging.DEBUG
logging.getLogger("docling").setLevel(logging_level)
logging.getLogger("PIL").setLevel(logging_level)
logging.getLogger("transformers").setLevel(logging_level)
logging.getLogger("datasets").setLevel(logging_level)
logging.getLogger("filelock").setLevel(logging_level)
logging.getLogger("urllib3").setLevel(logging_level)
logging.getLogger("docling_ibm_models").setLevel(logging_level)
logging.getLogger("matplotlib").setLevel(logging_level)

_log = logging.getLogger(__name__)

app = typer.Typer(
    name="docling-eval",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def derive_input_output_dirs(
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Path]:
    r"""
    One of the input or output dirs must be non None.
    In case one of them is None, it can be derived from the other one.
    """
    if input_dir and output_dir:
        return input_dir, output_dir
    if not input_dir and not output_dir:
        raise ValueError("Either input_dir or output_dir must be provided")

    if not input_dir and output_dir:
        # Derive input and output paths based on the directory structure in test_dataset_builder.py
        input_dir = output_dir / "eval_dataset" / benchmark.value / modality.value

    if not output_dir and input_dir:
        output_dir = input_dir.parent
    assert input_dir is not None
    assert output_dir is not None
    return input_dir, output_dir


def log_and_save_stats(
    odir: Path,
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    metric: str,
    stats: DatasetStatistics,
    log_filename: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    For the given DatasetStatistics, generate textual table and plot.

    Args:
        odir: Output directory
        benchmark: Benchmark name
        modality: Evaluation modality
        metric: Metric name
        stats: Dataset statistics
        log_filename: Optional log filename

    Returns:
        Tuple of (log_filename, fig_filename)
    """
    log_mode = "a"
    if log_filename is None:
        log_filename = (
            odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.txt"
        )
        log_mode = "w"
    fig_filename = odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.png"
    stats.save_histogram(figname=fig_filename, name=metric)

    data, headers = stats.to_table(metric)
    content = f"{benchmark.value} {modality.value} {metric}: "
    content += "mean={:.2f} median={:.2f} std={:.2f}\n\n".format(
        stats.mean, stats.median, stats.std
    )
    content += tabulate(data, headers=headers, tablefmt="github")
    content += "\n\n\n"

    _log.info(content)
    with open(log_filename, log_mode) as fd:
        fd.write(content)
        _log.info("Saving statistics report to %s", log_filename)

    return log_filename, fig_filename


def get_dataset_builder(
    benchmark: BenchMarkNames,
    target: Path,
    split: str = "test",
    begin_index: int = 0,
    end_index: int = -1,
    dataset_source: Optional[Path] = None,
    dataset_id: Optional[str] = None,
):
    """Get the appropriate dataset builder for the given benchmark."""
    common_params = {
        "target": target,
        "split": split,
        "begin_index": begin_index,
        "end_index": end_index,
    }

    if benchmark == BenchMarkNames.DPBENCH:
        return DPBenchDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCLING_DPBENCH:
        return DoclingDPBenchDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCLAYNETV1:
        return DocLayNetV1DatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCLAYNETV2:
        if dataset_source is None:
            raise ValueError("dataset_path is required for DocLayNetV2")
        return DocLayNetV2DatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.FUNSD:
        if dataset_source is None:
            raise ValueError("dataset_source is required for FUNSD")
        return FUNSDDatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.XFUND:
        if dataset_source is None:
            raise ValueError("dataset_source is required for XFUND")
        return XFUNDDatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.OMNIDOCBENCH:
        return OmniDocBenchDatasetBuilder(**common_params)  # type: ignore
    
    elif benchmark == BenchMarkNames.CUSTOM_OMNIDOCBENCH:
        return CustomOmniDocBenchDatasetBuilder(dataset_id=dataset_id, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.FINTABNET:
        return FintabNetDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.PUB1M:
        return PubTables1MDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.PUBTABNET:
        return PubTabNetDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCVQA:
        return DocVQADatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.CVAT:
        assert dataset_source is not None
        return CvatDatasetBuilder(
            name="CVAT", dataset_source=dataset_source, target=target, split=split
        )
    elif benchmark == BenchMarkNames.PLAIN_FILES:
        if dataset_source is None:
            raise ValueError("dataset_source is required for PLAIN_FILES")

        return FileDatasetBuilder(
            name=dataset_source.name,
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def get_prediction_provider(
    provider_type: PredictionProviderType,
    *,
    file_source_path: Optional[Path] = None,
    file_prediction_format: Optional[PredictionFormats] = None,
    file_use_ground_truth_images: bool = True,
    file_images_path: Optional[Path] = None,
    do_visualization: bool = True,
    do_table_structure: bool = True,
    artifacts_path: Optional[Path] = None,
    image_scale_factor: Optional[float] = None,
    docling_layout_model_spec: Optional[LayoutModelConfig] = None,
    docling_layout_create_orphan_clusters: Optional[bool] = None,
    docling_layout_keep_empty_clusters: Optional[bool] = None,
    # Controls orphan text cells only for the programmatic Docling pipeline (PDF_DOCLING)
    docling_programmatic_add_orphan_text_cells: Optional[bool] = None,
    docling_force_full_page_ocr: Optional[bool] = None,
):
    pipeline_options: PaginatedPipelineOptions
    """Get the appropriate prediction provider with default settings."""
    if (
        provider_type == PredictionProviderType.DOCLING
        or provider_type == PredictionProviderType.OCR_DOCLING
        or provider_type == PredictionProviderType.EasyOCR_DOCLING
    ):
        ocr_factory = get_ocr_factory()

        ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
            kind="easyocr",
            force_full_page_ocr=docling_force_full_page_ocr,
        )
        # Use all CPU cores
        accelerator_options = AcceleratorOptions(
            num_threads=multiprocessing.cpu_count(),
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
            do_table_structure=do_table_structure,
        )

        pipeline_options.images_scale = image_scale_factor or 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_parsed_pages = True
        pipeline_options.accelerator_options = accelerator_options

        # Layout options
        layout_options: LayoutOptions = LayoutOptions()
        if docling_layout_model_spec is not None:
            layout_options.model_spec = docling_layout_model_spec
        if docling_layout_create_orphan_clusters is not None:
            layout_options.create_orphan_clusters = (
                docling_layout_create_orphan_clusters
            )
        if docling_layout_keep_empty_clusters is not None:
            layout_options.keep_empty_clusters = docling_layout_keep_empty_clusters
        pipeline_options.layout_options = layout_options

        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            },
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.MacOCR_DOCLING:
        ocr_factory = get_ocr_factory()

        ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
            kind="ocrmac",
            force_full_page_ocr=docling_force_full_page_ocr,
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
            do_table_structure=do_table_structure,
        )

        pipeline_options.images_scale = image_scale_factor or 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            },
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.PDF_DOCLING:
        ocr_factory = get_ocr_factory()

        ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
            kind="easyocr",
            force_full_page_ocr=docling_force_full_page_ocr,
        )

        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            ocr_options=ocr_options,  # we need to provide OCR options in order to not break the parquet serialization
            do_table_structure=do_table_structure,
        )

        pdf_pipeline_options.images_scale = image_scale_factor or 2.0
        pdf_pipeline_options.generate_page_images = True
        pdf_pipeline_options.generate_picture_images = True

        # Only for programmatic Docling (PDF), optionally control orphan text cells
        if docling_programmatic_add_orphan_text_cells is not None:
            layout_options_prog = LayoutOptions()
            layout_options_prog.create_orphan_clusters = (
                docling_programmatic_add_orphan_text_cells
            )
            pdf_pipeline_options.layout_options = layout_options_prog

        ocr_pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,  # we need to provide OCR options in order to not break the parquet serialization
            do_table_structure=do_table_structure,
        )

        ocr_pipeline_options.images_scale = image_scale_factor or 2.0
        ocr_pipeline_options.generate_page_images = True
        ocr_pipeline_options.generate_picture_images = True

        if artifacts_path is not None:
            pdf_pipeline_options.artifacts_path = artifacts_path
            ocr_pipeline_options.artifacts_path = artifacts_path

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=ocr_pipeline_options
                ),
            },
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.SMOLDOCLING:
        pipeline_options = VlmPipelineOptions()

        pipeline_options.images_scale = image_scale_factor or 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        pipeline_options.vlm_options = smoldocling_vlm_conversion_options
        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        if sys.platform == "darwin":
            try:
                import mlx_vlm  # type: ignore

                pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options
                _log.info("running SmolDocling on MLX!")
            except ImportError:
                _log.warning(
                    "To run SmolDocling faster, please install mlx-vlm:\n"
                    "pip install mlx-vlm"
                )

        pdf_format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            },
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )
    elif provider_type == PredictionProviderType.GRANITEDOCLING:
        pipeline_options = VlmPipelineOptions()

        pipeline_options.images_scale = image_scale_factor or 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        pipeline_options.vlm_options = GRANITEDOCLING_TRANSFORMERS
        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        if sys.platform == "darwin":
            try:
                import mlx_vlm  # type: ignore

                pipeline_options.vlm_options = GRANITEDOCLING_MLX
                _log.info("running GraniteDocling on MLX!")
            except ImportError:
                _log.warning(
                    "To run SmolDocling faster, please install mlx-vlm:\n"
                    "pip install mlx-vlm"
                )

        pdf_format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            },
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )
    elif provider_type == PredictionProviderType.TABLEFORMER:
        return TableFormerPredictionProvider(
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )
    elif provider_type == PredictionProviderType.GOOGLE:
        return GoogleDocAIPredictionProvider(
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )
    elif provider_type == PredictionProviderType.AWS:
        return AWSTextractPredictionProvider(
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )
    elif provider_type == PredictionProviderType.AZURE:
        return AzureDocIntelligencePredictionProvider(
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.FILE:
        if file_prediction_format is None:
            raise ValueError("file_prediction_format is required for File provider")
        if file_source_path is None:
            raise ValueError("file_source_path is required for File provider")

        return FilePredictionProvider(
            prediction_format=file_prediction_format,
            source_path=file_source_path,
            do_visualization=do_visualization,
            ignore_missing_predictions=True,
            ignore_missing_files=True,
            use_ground_truth_page_images=file_use_ground_truth_images,
            prediction_images_path=file_images_path,
        )

    else:
        raise ValueError(f"Unsupported prediction provider: {provider_type}")


def evaluate(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
    cvat_overview_path: Optional[Path] = None,
) -> Optional[DatasetEvaluationType]:
    """Evaluate predictions against ground truth."""
    if not os.path.exists(idir):
        _log.error(f"Benchmark directory not found: {idir}")
        return None

    os.makedirs(odir, exist_ok=True)

    # Save the evaluation
    save_fn = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        _log.error("END2END evaluation not supported. ")
        return None

    elif modality == EvaluationModality.TIMINGS:
        timings_evaluator = TimingsEvaluator()
        evaluation = timings_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.LAYOUT:
        layout_evaluator = LayoutEvaluator(
            # missing_prediction_strategy=MissingPredictionStrategy.PENALIZE,
            # label_filtering_strategy=LabelFilteringStrategy.INTERSECTION,
            page_mapping_path=cvat_overview_path,
        )
        evaluation = layout_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.TABLE_STRUCTURE:
        table_evaluator = TableEvaluator()
        evaluation = table_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.DOCUMENT_STRUCTURE:
        doc_struct_evaluator = DocStructureEvaluator(
            page_mapping_path=cvat_overview_path
        )
        evaluation = doc_struct_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.OCR:
        if benchmark in [BenchMarkNames.XFUND, BenchMarkNames.PIXPARSEIDL]:
            text_unit = TextCellUnit.LINE
        else:
            text_unit = TextCellUnit.WORD

        logging.info(f"Benchmark received in evaluate: {benchmark} ({type(benchmark)})")
        logging.info(f"Text unit set to {text_unit}")

        ocr_evaluator = OCREvaluator(
            intermediate_evaluations_path=odir, text_unit=text_unit
        )
        evaluation = ocr_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.READING_ORDER:
        readingorder_evaluator = ReadingOrderEvaluator()
        evaluation = readingorder_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(
                evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        md_evaluator = MarkdownTextEvaluator()
        evaluation = md_evaluator(  # type: ignore
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(
                evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.BBOXES_TEXT:
        bbox_evaluator = BboxTextEvaluator()
        evaluation = bbox_evaluator(  # type: ignore
            idir,
            split=split,
        )
        with open(save_fn, "w") as fd:
            json.dump(
                evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.KEY_VALUE:
        keyvalue_evaluator = KeyValueEvaluator(page_mapping_path=cvat_overview_path)
        evaluation = keyvalue_evaluator(  # type: ignore
            idir,
            split=split,
        )
        with open(save_fn, "w") as fd:
            json.dump(
                evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    else:
        _log.error(f"Unsupported modality for evaluation: {modality}")
        return None

    _log.info(f"The evaluation has been saved in '{save_fn}'")
    return evaluation  # type: ignore


def visualize(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
):
    """
    Visualize evaluation results.

    Args:
        modality: Visualization modality
        benchmark: Benchmark name
        idir: Input directory with dataset
        odir: Output directory for visualizations
        split: Dataset split
        begin_index: Begin index
        end_index: End index
    """
    if not os.path.exists(idir):
        _log.error(f"Input directory not found: {idir}")
        return

    os.makedirs(odir, exist_ok=True)
    metrics_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if not os.path.exists(metrics_filename):
        _log.error(f"Metrics file not found: {metrics_filename}")
        _log.error("You need to run evaluation first before visualization")
        return

    if modality == EvaluationModality.END2END:
        _log.error("END2END visualization not supported")

    elif modality == EvaluationModality.TIMINGS:
        try:
            with open(metrics_filename, "r") as fd:
                timings_evaluation = DatasetTimingsEvaluation.model_validate_json(
                    fd.read()
                )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "time_to_solution_per_doc",
                timings_evaluation.timing_per_document_stats,
            )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "time_to_solution_per_page",
                timings_evaluation.timing_per_page_stats,
            )
        except Exception as e:
            _log.error(f"Error processing timings evaluation: {str(e)}")

    elif modality == EvaluationModality.LAYOUT:
        try:
            with open(metrics_filename, "r") as fd:
                layout_evaluation = DatasetLayoutEvaluation.model_validate_json(
                    fd.read()
                )

            # Save layout statistics for mAP
            log_filename, _ = log_and_save_stats(
                odir,
                benchmark,
                modality,
                "mAP_0.5_0.95",
                layout_evaluation.map_stats,
            )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "precision",
                layout_evaluation.segmentation_precision_stats,
            )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "recall",
                layout_evaluation.segmentation_recall_stats,
            )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "f1",
                layout_evaluation.segmentation_f1_stats,
            )

            # Append to layout statistics, the AP per classes
            data, headers = layout_evaluation.to_table()
            content = "\n\n\nAP[0.5:0.05:0.95] per class (reported as %):\n\n"
            content += tabulate(data, headers=headers, tablefmt="github")

            # Append to layout statistics, the mAP
            content += "\n\nTotal mAP[0.5:0.05:0.95] (reported as %): {:.2f}".format(
                100.0 * layout_evaluation.mAP
            )
            _log.info(content)
            with open(log_filename, "a") as fd:
                fd.write(content)
        except Exception as e:
            _log.error(f"Error processing layout evaluation: {str(e)}")

    elif modality == EvaluationModality.TABLE_STRUCTURE:
        try:
            with open(metrics_filename, "r") as fd:
                table_evaluation = DatasetTableEvaluation.model_validate_json(fd.read())

            figname = (
                odir
                / f"evaluation_{benchmark.value}_{modality.value}-delta_row_col.png"
            )
            table_evaluation.save_histogram_delta_row_col(figname=figname)

            # TEDS struct-with-text
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "TEDS_struct-with-text",
                table_evaluation.TEDS,
            )

            # TEDS struct-only
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "TEDS_struct-only",
                table_evaluation.TEDS_struct,
            )
        except Exception as e:
            _log.error(f"Error processing table evaluation: {str(e)}")

    elif modality == EvaluationModality.READING_ORDER:
        try:
            with open(metrics_filename, "r") as fd:
                ro_evaluation = DatasetReadingOrderEvaluation.model_validate_json(
                    fd.read()
                )

            # ARD
            log_and_save_stats(
                odir, benchmark, modality, "ARD_norm", ro_evaluation.ard_stats
            )

            # Weighted ARD
            log_and_save_stats(
                odir, benchmark, modality, "weighted_ARD", ro_evaluation.w_ard_stats
            )

            # Generate visualizations of the reading order across the GT and the prediction
            ro_visualizer = ReadingOrderVisualizer()
            ro_visualizer(
                idir,
                metrics_filename,
                odir,
                split=split,
            )
        except Exception as e:
            _log.error(f"Error processing reading order evaluation: {str(e)}")

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        try:
            with open(metrics_filename, "r") as fd:
                md_evaluation = DatasetMarkdownEvaluation.model_validate_json(fd.read())

            # Log stats for all metrics in the same file
            log_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.txt"
            with open(log_filename, "w") as fd:
                fd.write(
                    f"{benchmark.value} size: {len(md_evaluation.evaluations)}\n\n"
                )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "BLEU",
                md_evaluation.bleu_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "F1",
                md_evaluation.f1_score_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "precision",
                md_evaluation.precision_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "recall",
                md_evaluation.recall_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "edit_distance",
                md_evaluation.edit_distance_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "meteor",
                md_evaluation.meteor_stats,
                log_filename=log_filename,
            )
        except Exception as e:
            _log.error(f"Error processing markdown text evaluation: {str(e)}")

    elif modality == EvaluationModality.OCR:
        try:
            with open(metrics_filename, "r") as fd:
                ocr_evaluation = OcrDatasetEvaluationResult.model_validate_json(
                    fd.read()
                )

            log_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.txt"
            with open(log_filename, "w") as fd:
                fd.write(f"{benchmark.value}\n\n")
                fd.write(f"F1 Score: {ocr_evaluation.f1_score:.2f}\n")
                fd.write(f"Recall: {ocr_evaluation.recall:.2f}\n")
                fd.write(f"Precision: {ocr_evaluation.precision:.2f}\n")

            _log.info(f"OCR evaluation stats saved to {log_filename}")

            ocr_visualizer = OCRVisualizer()
            ocr_visualizer(
                dataset_path=idir,
                ocr_evaluation_report_path=metrics_filename,
                output_directory=odir,
                data_split_name=split,
            )
        except Exception as e:
            _log.error(f"Error processing OCR evaluation: {str(e)}")

    else:
        _log.error(f"Unsupported modality for visualization: {modality}")


@app.command()
def create_sliced_pdfs(
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    source_dir: Annotated[Path, typer.Option(help="Dataset source path with PDFs")],
    slice_length: Annotated[int, typer.Option(help="sliding window")] = 1,
    num_overlap: Annotated[int, typer.Option(help="overlap window")] = 0,
):
    """Process multi-page pdf documents into chunks of slice_length with num_overlap overlapping pages in each slice."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if slice_length < 1:
        return ValueError("slice-length must be at least 1.")
    if num_overlap > slice_length - 1:
        return ValueError("num-overlap must be at most one less than slice-length")

    num_overlap = max(num_overlap, 0)

    pdf_paths = glob.glob(f"{source_dir}/**/*.pdf", recursive=True)
    _log.info(f"#-pdfs: {pdf_paths}")

    for pdf_path in pdf_paths:
        base_name = os.path.basename(pdf_path).replace(".pdf", "")

        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                total_pages = len(reader.pages)

                _log.info(f"Processing {pdf_path} ({total_pages} pages)")

                for start_page in range(0, total_pages, slice_length - num_overlap):
                    end_page = min(start_page + slice_length, total_pages)

                    # Create a new PDF with the pages in the current window
                    writer = PdfWriter()

                    for page_num in range(start_page, end_page):
                        writer.add_page(reader.pages[page_num])

                    # Save the new PDF
                    output_path = os.path.join(
                        output_dir, f"{base_name}_ps_{start_page}_pe_{end_page}.pdf"
                    )
                    with open(output_path, "wb") as output_file:
                        writer.write(output_file)

        except Exception as e:
            _log.error(f"Error processing {pdf_path}: {e}")


@app.command()
def create_cvat(
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    gt_dir: Annotated[Path, typer.Option(help="Dataset source path")],
    bucket_size: Annotated[int, typer.Option(help="Size of CVAT tasks")] = 20,
    use_predictions: Annotated[bool, typer.Option(help="use predictions")] = False,
    sliding_window: Annotated[
        int,
        typer.Option(
            help="Size of sliding window for page processing (1 for single pages, >1 for multi-page windows)"
        ),
    ] = 2,
):
    """Create dataset ready to upload to CVAT starting from (ground-truth) dataset."""
    builder = CvatPreannotationBuilder(
        dataset_source=gt_dir,
        target=output_dir,
        bucket_size=bucket_size,
        use_predictions=use_predictions,
        sliding_window=sliding_window,
    )
    builder.prepare_for_annotation()


@app.command()
def create_gt(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    dataset_source: Annotated[
        Optional[Path], typer.Option(help="Dataset source path")
    ] = None,
    dataset_id: Annotated[
        Optional[str], typer.Option(help="Dataset ID for custom OmniDocBench")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = 80,
    do_visualization: Annotated[
        bool, typer.Option(help="visualize the predictions")
    ] = True,
):
    """Create ground truth dataset only."""
    gt_dir = output_dir / "gt_dataset"

    try:
        dataset_builder = get_dataset_builder(
            benchmark=benchmark,
            target=gt_dir,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
            dataset_source=dataset_source,
            dataset_id=dataset_id,
        )

        # Retrieve and save the dataset
        if dataset_builder.must_retrieve:
            dataset_builder.retrieve_input_dataset()
        dataset_builder.save_to_disk(
            chunk_size=chunk_size, do_visualization=do_visualization
        )

        _log.info(f"Ground truth dataset created at {gt_dir}")
    except ValueError as e:
        _log.error(f"Error creating dataset builder: {str(e)}")


@app.command()
def create_eval(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory.")],
    prediction_provider: Annotated[
        PredictionProviderType, typer.Option(help="Type of prediction provider to use")
    ],
    gt_dir: Annotated[
        Optional[Path], typer.Option(help="Input directory for GT")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = 80,
    # File provider required options
    file_prediction_format: Annotated[
        Optional[str],
        typer.Option(
            help="Prediction format for File provider (required if using FILE provider)"
        ),
    ] = None,
    file_source_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Source path for prediction files (required if using FILE provider)"
        ),
    ] = None,
    file_use_ground_truth_images: Annotated[
        bool,
        typer.Option(
            help="Use the GT images to construct the prediction dataset (if using FILE provider)"
        ),
    ] = True,
    file_images_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Source path for the prediction images (if using FILE provider)"
        ),
    ] = None,
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for local model artifacts. Will only be passed to providers supporting this."
        ),
    ] = None,
    docling_layout_model_spec: Annotated[
        Optional[str],
        typer.Option(
            help="Layout model spec for Docling. Supported values: {}".format(
                DoclingLayoutOptionsManager.get_layout_model_config_names()
            )
        ),
    ] = "docling_layout_heron",
    docling_layout_create_orphan_clusters: Annotated[
        Optional[bool],
        typer.Option(
            help="Enable orphan clusters creation in Docling layout post-processing"
        ),
    ] = True,
    docling_layout_keep_empty_clusters: Annotated[
        Optional[bool],
        typer.Option(help="Keep the empty clusters in Docling layout post-processing"),
    ] = False,
    programmatic_add_orphan_text_cells: Annotated[
        bool,
        typer.Option(
            help=(
                "Add orphan text cells for programmatic Docling pipeline (PDF_DOCLING). "
                "Defaults to False."
            )
        ),
    ] = False,
    do_visualization: Annotated[
        bool, typer.Option(help="visualize the predictions")
    ] = True,
    image_scale_factor: Annotated[
        float,
        typer.Option(help="Scale of page images used in prediction (only Docling)"),
    ] = 2.0,
    do_table_structure: Annotated[
        bool, typer.Option(help="Include table structure predictions (only Docling)")
    ] = True,
    docling_force_full_page_ocr: Annotated[
        bool,
        typer.Option(help="Force OCR on entire page (only Docling OCR providers)"),
    ] = False,
):
    """Create evaluation dataset from existing ground truth."""
    gt_dir = gt_dir or output_dir / "gt_dataset"
    pred_dir = output_dir / "eval_dataset"

    # Check if ground truth exists
    if not gt_dir.exists():
        _log.error(f"Ground truth directory not found: {gt_dir}")
        _log.error(
            "Cannot create eval dataset without ground truth. Run create_gt first."
        )
        return

    try:
        # Convert string option to enum value
        file_format = (
            PredictionFormats(file_prediction_format)
            if file_prediction_format
            else None
        )

        # Create the appropriate prediction provider
        docling_layout_model_spec_obj = (
            DoclingLayoutOptionsManager.get_layout_model_config(
                docling_layout_model_spec
            )
            if docling_layout_model_spec
            else None
        )

        provider = get_prediction_provider(
            provider_type=prediction_provider,
            file_source_path=file_source_path,
            file_prediction_format=file_format,
            file_use_ground_truth_images=file_use_ground_truth_images,
            file_images_path=file_images_path,
            artifacts_path=artifacts_path,
            do_visualization=do_visualization,
            image_scale_factor=image_scale_factor,
            do_table_structure=do_table_structure,
            docling_layout_model_spec=docling_layout_model_spec_obj,
            docling_layout_create_orphan_clusters=docling_layout_create_orphan_clusters,
            docling_layout_keep_empty_clusters=docling_layout_keep_empty_clusters,
            docling_programmatic_add_orphan_text_cells=programmatic_add_orphan_text_cells,
            docling_force_full_page_ocr=docling_force_full_page_ocr,
        )

        # Get the dataset name from the benchmark
        dataset_name = f"{benchmark.value}"

        # Create predictions
        provider.create_prediction_dataset(
            name=dataset_name,
            gt_dataset_dir=gt_dir,
            target_dataset_dir=pred_dir,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
            chunk_size=chunk_size,
        )

        _log.info(f"Evaluation dataset created at {pred_dir}")
    except ValueError as e:
        _log.error(f"Error creating prediction provider: {str(e)}")


@app.command()
def create(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    dataset_source: Annotated[
        Optional[Path], typer.Option(help="Dataset source path")
    ] = None,
    dataset_id: Annotated[
        Optional[str], typer.Option(help="Dataset ID for custom OmniDocBench")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = 80,
    prediction_provider: Annotated[
        Optional[PredictionProviderType],
        typer.Option(help="Type of prediction provider to use"),
    ] = None,
    file_prediction_format: Annotated[
        Optional[str], typer.Option(help="Prediction format for File provider")
    ] = None,
    file_source_path: Annotated[
        Optional[Path], typer.Option(help="Source path for File provider")
    ] = None,
    do_visualization: Annotated[
        bool, typer.Option(help="visualize the predictions")
    ] = True,
    image_scale_factor: Annotated[
        float,
        typer.Option(help="Scale of page images used in prediction (only Docling)"),
    ] = 2.0,
    do_table_structure: Annotated[
        bool, typer.Option(help="Include table structure predictions (only Docling)")
    ] = True,
    docling_force_full_page_ocr: Annotated[
        bool,
        typer.Option(help="Force OCR on entire page (only Docling OCR providers)"),
    ] = False,
    programmatic_add_orphan_text_cells: Annotated[
        bool,
        typer.Option(
            help=(
                "Add orphan text cells for programmatic Docling pipeline (PDF_DOCLING). "
                "Defaults to False."
            )
        ),
    ] = False,
):
    """Create both ground truth and evaluation datasets in one step."""
    # First create ground truth
    create_gt(
        benchmark=benchmark,
        output_dir=output_dir,
        dataset_source=dataset_source,
        dataset_id=dataset_id,
        split=split,
        begin_index=begin_index,
        end_index=end_index,
        chunk_size=chunk_size,
        do_visualization=do_visualization,
    )

    # Then create evaluation if provider specified
    if prediction_provider:
        create_eval(
            benchmark=benchmark,
            output_dir=output_dir,
            prediction_provider=prediction_provider,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
            chunk_size=chunk_size,
            file_prediction_format=file_prediction_format,
            file_source_path=file_source_path,
            do_visualization=do_visualization,
            image_scale_factor=image_scale_factor,
            do_table_structure=do_table_structure,
            docling_force_full_page_ocr=docling_force_full_page_ocr,
            programmatic_add_orphan_text_cells=programmatic_add_orphan_text_cells,
        )
    else:
        _log.info(
            "No prediction provider specified, skipping evaluation dataset creation"
        )


@app.command(name="evaluate")
def evaluate_cmd(
    modality: Annotated[EvaluationModality, typer.Option(help="Evaluation modality")],
    benchmark: Annotated[
        BenchMarkNames,
        typer.Option(
            help="Benchmark name. It is used only to set the filename of the evaluation json file."
        ),
    ],
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory with evaluation dataset. If not provided, the input directory will be derived from the output directory."
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Base output directory. If not provided, the output directory will be derived from the input directory."
        ),
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
):
    """Evaluate predictions against ground truth."""
    input_dir, output_dir = derive_input_output_dirs(
        benchmark, modality, input_dir, output_dir
    )
    assert input_dir is not None
    assert output_dir is not None
    eval_output_dir = output_dir / "evaluations" / modality.value

    # Create output directory
    os.makedirs(eval_output_dir, exist_ok=True)

    # Call our self-contained evaluation function
    evaluate(
        modality=modality,
        benchmark=benchmark,
        idir=input_dir,
        odir=eval_output_dir,
        split=split,
    )


@app.command(name="visualize")
def visualize_cmd(
    modality: Annotated[
        EvaluationModality, typer.Option(help="Visualization modality")
    ],
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory with evaluation dataset. If not provided, the input directory will be derived from the output directory."
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Base output directory. If not provided, the output directory will be derived from the input directory."
        ),
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
):
    """Visualize evaluation results."""
    input_dir, output_dir = derive_input_output_dirs(
        benchmark, modality, input_dir, output_dir
    )
    assert input_dir is not None
    assert output_dir is not None
    eval_output_dir = output_dir / "evaluations" / modality.value

    # Create output directory
    os.makedirs(eval_output_dir, exist_ok=True)

    # Call our self-contained visualization function
    visualize(
        modality=modality,
        benchmark=benchmark,
        idir=input_dir,
        odir=eval_output_dir,
        split=split,
    )


@app.callback()
def main():
    """Docling Evaluation CLI for benchmarking document processing tasks."""
    pass


if __name__ == "__main__":
    app()
