import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    PageItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.benchmarks.utils import get_binary, get_binhash
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
)
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)

TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class PixparseDatasetBuilder(BaseEvaluationDatasetBuilder):
    def __init__(
        self,
        dataset_source: Path,
        prediction_provider: BasePredictionProvider,
        target: Path,
        do_visualization: bool = True,
        split: str = "test",
        max_items: int = -1,
    ):
        super().__init__(
            name="pixparse",
            dataset_source=dataset_source,
            prediction_provider=prediction_provider,
            target=target,
        )
        self.dataset_local_path = dataset_source
        self.do_visualization = do_visualization
        self.split = split
        self.max_items = max_items

    def _create_ground_truth_doc(
        self, doc_id: str, gt_data: Dict, image_file: Path
    ) -> DoclingDocument:
        """Create a DoclingDocument from ground truth data and image file."""
        w, h = Image.open(image_file).size

        true_doc = DoclingDocument(name=doc_id)
        true_doc.pages[1] = PageItem(
            size=Size(width=float(w), height=float(h)), page_no=1
        )

        for page_idx, page in enumerate(gt_data["pages"], 1):
            for text, bbox, _ in zip(page["text"], page["bbox"], page["score"]):
                bbox_obj = BoundingBox.from_tuple(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[0] + bbox[2]),
                        float(bbox[1] + bbox[3]),
                    ),
                    CoordOrigin.TOPLEFT,
                )
                prov = ProvenanceItem(
                    page_no=page_idx, bbox=bbox_obj, charspan=(0, len(text))
                )
                true_doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

        return true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        output_dir = self.target
        output_dir.mkdir(parents=True, exist_ok=True)

        viz_dir = output_dir / "visualizations"
        if self.do_visualization:
            viz_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_files = list(self.dataset_local_path.rglob("ground_truth.json"))
        if self.max_items > 0:
            ground_truth_files = ground_truth_files[: self.max_items]

        for gt_file in tqdm(
            ground_truth_files,
            desc="Processing files for OCR Benchmark",
            total=len(ground_truth_files),
            ncols=128,
        ):
            try:
                image_file = gt_file.parent / "original.tif"
                if not image_file.exists():
                    logging.info(f"Warning: No image file found for {gt_file}")
                    continue

                doc_id = gt_file.parent.name
                input_dir = self.dataset_local_path

                with open(gt_file, "r") as f:
                    gt_data = json.load(f)

                true_doc = self._create_ground_truth_doc(doc_id, gt_data, image_file)

                image_bytes = get_binary(image_file)
                image_stream = DocumentStream(
                    name=image_file.name, stream=BytesIO(image_bytes)
                )
                record = DatasetRecord(
                    predictor_info=self.prediction_provider.info(),
                    doc_id=doc_id,
                    doc_hash=get_binhash(image_bytes),
                    ground_truth_doc=true_doc,
                    original=image_stream,
                    mime_type="image/tiff",
                )

                self.update_prediction(record=record)

                # Generate visualization if requested
                if self.do_visualization and record.predicted_doc is not None:
                    image: Image.Image = Image.open(BytesIO(image_bytes))
                    if image.mode not in (
                        "RGB",
                        "RGBA",
                        "L",
                    ):
                        image = image.convert("RGB")
                    save_comparison_html_with_clusters(
                        filename=viz_dir / f"{doc_id}-clusters.html",
                        true_doc=true_doc,
                        pred_doc=record.predicted_doc,
                        page_image=image,
                        true_labels=TRUE_HTML_EXPORT_LABELS,
                        pred_labels=PRED_HTML_EXPORT_LABELS,
                    )
                yield record

            except Exception as e:
                logging.error(f"Error processing {gt_file}: {str(e)}")
                raise
