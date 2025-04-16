import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.legacy.cvat_annotation.utils import (
    AnnotatedDoc,
    AnnotatedImage,
    AnnotationBBox,
    AnnotationOverview,
    BenchMarkDirs,
    DocLinkLabel,
    TableComponentLabel,
    rgb_to_hex,
)
from docling_eval.utils.utils import get_binhash, insert_images_from_pil

# Configure logging
_log = logging.getLogger(__name__)


class CvatPreannotationBuilder:
    """
    Builder class for creating CVAT preannotations from a dataset.

    This class takes an existing dataset (ground truth or with predictions)
    and prepares files and preannotations for CVAT annotation.
    """

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        bucket_size: int = 200,
    ):
        """
        Initialize the CvatPreannotationBuilder.

        Args:
            source_dir: Directory containing the source dataset
            target_dir: Directory where CVAT preannotations will be saved
            bucket_size: Number of documents per bucket for CVAT tasks
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.bucket_size = bucket_size
        self.benchmark_dirs = BenchMarkDirs()
        self.benchmark_dirs.set_up_directory_structure(
            source=source_dir, target=target_dir
        )
        self.overview = AnnotationOverview()

    def export_from_dataset(self) -> AnnotationOverview:
        """
        Export supplementary files from the dataset.

        Returns:
            AnnotationOverview object with dataset information
        """
        test_files = sorted(
            glob.glob(str(self.benchmark_dirs.source_dir / "*.parquet"))
        )
        ds = load_dataset("parquet", data_files={"test": test_files})

        if ds is None:
            raise ValueError(
                f"Failed to load dataset from {self.benchmark_dirs.source_dir}"
            )

        ds_selection = ds["test"]
        overview = AnnotationOverview()

        for data in ds_selection:

            dataset_record = None
            try:
                data_record = DatasetRecordWithPrediction.model_validate(data)
            except:
                data_record = DatasetRecord.model_validate(data)

            # Use document ID as name
            doc_name = f"{data_record.doc_id}"

            bin_doc = data_record.original
            doc_hash = data_record.doc_hash

            if doc_hash is None and bin_doc is not None:
                doc_hash = get_binhash(binary_data=bin_doc.stream.read())
                # Reset stream position after reading
                bin_doc.stream.seek(0)

            # Write ground truth and predicted document
            true_file = self.benchmark_dirs.json_true_dir / f"{doc_name}.json"
            true_doc = data_record.ground_truth_doc
            true_doc = insert_images_from_pil(
                true_doc,
                data_record.ground_truth_pictures,
                data_record.ground_truth_page_images,
            )
            true_doc.save_as_json(filename=true_file)

            pred_file = true_file
            if isinstance(dataset_record, DatasetRecordWithPrediction):
                # Process predicted document if available
                pred_file = self.benchmark_dirs.json_pred_dir / f"{doc_name}.json"
                pred_doc = data_record.predicted_doc
                if pred_doc is not None:
                    pred_doc = insert_images_from_pil(
                        pred_doc,
                        data_record.predicted_pictures,
                        data_record.predicted_page_images,
                    )
                    pred_doc.save_as_json(filename=pred_file)

            mime_type = data_record.mime_type

            # Determine file extension based on MIME type
            bin_name = None
            if mime_type == "application/pdf":
                bin_name = f"{doc_hash}.pdf"
            elif mime_type == "image/png":
                bin_name = f"{doc_hash}.png"
            elif mime_type == "image/jpg" or mime_type == "image/jpeg":
                bin_name = f"{doc_hash}.jpg"
            else:
                raise ValueError(f"Unsupported mime-type {mime_type}")

            # Write binary document
            bin_file = self.benchmark_dirs.bins_dir / bin_name
            if bin_doc is not None:
                with open(bin_file, "wb") as fw:
                    fw.write(bin_doc.stream.read())
                    # Reset stream position after writing
                    bin_doc.stream.seek(0)

            overview.doc_annotations.append(
                AnnotatedDoc(
                    mime_type=mime_type,
                    true_file=true_file,
                    pred_file=pred_file,
                    bin_file=bin_file,
                    doc_hash=doc_hash,
                    doc_name=doc_name,
                )
            )

        return overview

    def create_project_properties(self) -> None:
        """
        Create CVAT project properties file.
        """
        results = []

        # Add DocItemLabel properties
        for item in DocItemLabel:
            r, g, b = DocItemLabel.get_color(item)

            results.append(
                {
                    "name": item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "rectangle",
                    "attributes": [],
                }
            )

            # Add specific attributes for certain labels
            if item in [DocItemLabel.LIST_ITEM, DocItemLabel.SECTION_HEADER]:
                results[-1]["attributes"].append(
                    {
                        "name": "level",
                        "input_type": "number",
                        "mutable": True,
                        "values": ["1", "10", "1"],
                        "default_value": "1",
                    }
                )

            if item == DocItemLabel.FORMULA:
                results[-1]["attributes"].append(
                    {
                        "name": "latex",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

            if item == DocItemLabel.CODE:
                results[-1]["attributes"].append(
                    {
                        "name": "code",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

            if item == DocItemLabel.PICTURE:
                results[-1]["attributes"].append(
                    {
                        "name": "json",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

        # Add TableComponentLabel properties
        for table_item in TableComponentLabel:
            r, g, b = TableComponentLabel.get_color(table_item)

            results.append(
                {
                    "name": table_item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "rectangle",
                    "attributes": [],
                }
            )

        # Add DocLinkLabel properties
        for link_item in DocLinkLabel:
            r, g, b = DocLinkLabel.get_color(link_item)

            results.append(
                {
                    "name": link_item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "polyline",
                    "attributes": [],
                }
            )

        _log.info(
            f"Writing project description: {str(self.benchmark_dirs.project_desc_file)}"
        )
        with open(str(self.benchmark_dirs.project_desc_file), "w") as fw:
            json.dump(results, fw, indent=2)

    def create_preannotation_files(self) -> None:
        """
        Create CVAT preannotation files.
        """
        cvat_annots: List[str] = []

        img_id, img_cnt, bucket_id = 0, 0, 0
        for doc_overview in self.overview.doc_annotations:
            try:
                doc = DoclingDocument.load_from_json(doc_overview.pred_file)
            except Exception as e:
                _log.error(f"Failed to load document {doc_overview.pred_file}: {e}")
                continue

            for page_no, page in doc.pages.items():
                img_cnt += 1

                bucket_id = int((img_cnt - 1) / float(self.bucket_size))
                bucket_dir = self.benchmark_dirs.tasks_dir / f"task_{bucket_id:02}"

                if not os.path.exists(bucket_dir) and len(cvat_annots) > 0:
                    # Write the pre-annotation files
                    _log.info(f"#-annots: {len(cvat_annots)}")

                    prev_bucket_id = int((img_cnt - 2) / float(self.bucket_size))
                    preannot_file = (
                        self.benchmark_dirs.tasks_dir
                        / f"task_{prev_bucket_id:02}_preannotate.xml"
                    )

                    with open(preannot_file, "w") as fw:
                        fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
                        fw.write("<annotations>\n")
                        for cvat_annot in cvat_annots:
                            fw.write(f"{cvat_annot}\n")
                        fw.write("</annotations>\n")

                    cvat_annots = []

                os.makedirs(bucket_dir, exist_ok=True)

                doc_name = doc_overview.doc_name
                doc_hash = doc_overview.doc_hash

                filename = f"doc_{doc_hash}_page_{page_no:06}.png"

                annotated_image = AnnotatedImage(
                    img_id=img_cnt,
                    mime_type=doc_overview.mime_type,
                    true_file=doc_overview.true_file,
                    pred_file=doc_overview.pred_file,
                    bin_file=doc_overview.bin_file,
                    doc_name=doc_name,
                    doc_hash=doc_hash,
                    bucket_dir=bucket_dir,
                    filename=filename,
                )

                annotated_image.img_file = bucket_dir / filename

                page_img_file = self.benchmark_dirs.page_imgs_dir / filename
                annotated_image.page_img_files = [page_img_file]

                page_image_ref = page.image
                if page_image_ref is not None:
                    page_image = page_image_ref.pil_image

                    if page_image is not None:
                        page_image.save(str(annotated_image.img_file))
                        page_image.save(str(annotated_image.page_img_files[0]))

                        annotated_image.img_w = page_image.width
                        annotated_image.img_h = page_image.height

                        annotated_image.page_nos = [page_no]
                        self.overview.img_annotations[filename] = annotated_image
                    else:
                        _log.warning(
                            f"Missing pillow image of page {page_no}, skipping..."
                        )
                        continue
                else:
                    _log.warning(f"Missing image-ref of page {page_no}, skipping...")
                    continue

                # Extract bounding boxes for annotation
                page_bboxes = []
                for item, _ in doc.iterate_items():
                    for prov in item.prov:
                        if page_no == prov.page_no:
                            page_w = doc.pages[prov.page_no].size.width
                            page_h = doc.pages[prov.page_no].size.height

                            img_w = annotated_image.img_w
                            img_h = annotated_image.img_h

                            page_bbox = prov.bbox.to_top_left_origin(page_height=page_h)

                            page_bboxes.append(
                                AnnotationBBox(
                                    bbox_id=len(page_bboxes),
                                    label=item.label,
                                    bbox=BoundingBox(
                                        l=page_bbox.l / page_w * img_w,
                                        r=page_bbox.r / page_w * img_w,
                                        t=page_bbox.t / page_h * img_h,
                                        b=page_bbox.b / page_h * img_h,
                                        coord_origin=page_bbox.coord_origin,
                                    ),
                                )
                            )

                annotated_image.pred_boxes = page_bboxes
                cvat_annots.append(annotated_image.to_cvat())

        # Write remaining preannotation files
        if len(cvat_annots) > 0:
            preannot_file = (
                self.benchmark_dirs.tasks_dir / f"task_{bucket_id:02}_preannotate.xml"
            )

            with open(preannot_file, "w") as fw:
                fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
                fw.write("<annotations>\n")
                for cvat_annot in cvat_annots:
                    fw.write(f"{cvat_annot}\n")
                fw.write("</annotations>\n")

        # Save overview
        self.overview.save_as_json(self.benchmark_dirs.overview_file)

    def prepare_for_annotation(self) -> None:
        """
        Prepare all necessary files for CVAT annotation.
        """
        self.create_project_properties()
        self.overview = self.export_from_dataset()
        self.create_preannotation_files()
