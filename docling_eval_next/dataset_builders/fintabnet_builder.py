import glob
import io
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from PIL.Image import Image
from tqdm import tqdm

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binhash,
)
from docling_eval.converters.models.tableformer.tf_model_prediction import PageTokens
from docling_eval.visualisation.visualisations import save_comparison_html
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
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
    # Additional
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
    # Additional
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}



class FintabnetDatasetBuilder(BaseEvaluationDatasetBuilder):
    """ Base Fintabnet Dataset Builder that will pull dataset from Hugging face."""
    def __init__(
        self,
        name: str,
        prediction_provider: BasePredictionProvider,
        target: Path,
        do_visualization: bool = True,
    ):
        super().__init__(
            name=name,
            dataset_source=HFSource(repo_id="ds4sd/FinTabNet_OTSL"),
            prediction_provider=prediction_provider,
            target=target,
        )
        self.do_visualization = do_visualization


class FintabnetTableStructureDatasetBuilder(FintabnetDatasetBuilder):
    """ Subclass of FintabnetDatasetBuilder that will define the "iterate" method on how to iterate
    the table structure from the dataset."""
    def __init__(
        self,
        prediction_provider: BasePredictionProvider,
        target: Path,
        do_visualization: bool = True,
    ):
        super().__init__(
            name="Fintabnet: table-structure",
            prediction_provider=prediction_provider,
            target=target,
            do_visualization=do_visualization,
        )

    def create_page_tokens(self, data: List[Any], height: float, width: float) -> PageTokens:
        """Needed for tableformer model only, where it additionally needs the page tokens for extraction.
        TODO: Not needed?? - Remove for hyperscalers"""
        tokens = []
        # text_lines = []

        cnt = 0
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                text = "".join(item["tokens"])

                tokens.append(
                    {
                        "bbox": {
                            "l": item["bbox"][0],
                            # "t": height - item["bbox"][3],
                            "t": item["bbox"][1],
                            "r": item["bbox"][2],
                            # "b": height - item["bbox"][1],
                            "b": item["bbox"][3],
                            # "coord_origin": str(CoordOrigin.BOTTOMLEFT.value)
                            "coord_origin": str(CoordOrigin.TOPLEFT.value),
                        },
                        "text": "".join(item["tokens"]),
                        "id": cnt,
                    }
                )
                cnt += 1

        result = {"tokens": tokens, "height": height, "width": width}
        return PageTokens.parse_obj(result)

    def iterate(self) -> Iterable[DatasetRecord]:
        """Iterate and yield each record of the dataset.
        Prediction will be run on the yielded record in the calling function."""

        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        # Create the folders for saving the intermediate files (test) and visualizations
        intermediate_dir = self.target / "intermediate_files"
        viz_dir = self.target / "vizualisations"
        for _ in [intermediate_dir, viz_dir]:
            os.makedirs(_, exist_ok=True)

        # Use glob to find all .parquet files in the directory and clean up the intermediate files
        parquet_files = glob.glob(os.path.join(str(intermediate_dir), "*.parquet"))
        for file in parquet_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        print(f"self.dataset_local_path={self.dataset_local_path}")
        print(f"self.name={self.name}")
        ds = load_dataset(os.path.join(self.dataset_local_path, "data"), split="test") # TODO - pass the split as argument?

        # TODO - Pass this as an argument? Do we need to run all items..
        max_items = -1
        if max_items == -1:
            max_items = len(ds)

        # Iterate each of the record in the dataset
        for i, item in tqdm(
            enumerate(ds),
            total=max_items,
            ncols=128,
            desc=f"create Fintabnet dataset",
        ):
            filename = item["filename"]
            table_image = item["image"]

            # TODO - For now, process two files instead of the whole dataset
            if filename not in ["HAL.2015.page_43.pdf_125177.png", "HAL.2009.page_77.pdf_125051.png"]:
                continue

            print(f"\nProcessing file - [{filename}]...")

            true_page_images = [table_image]
            # page_tokens = self.create_page_tokens(
            #     data=item["cells"], height=table_image.height, width=table_image.width
            # )

            # Create the Ground truth document
            true_doc = DoclingDocument(name=f"ground-truth {filename}")

            page_index = 1

            image_scale = 1.0 # TODO - pass as input argument?

            image_ref = ImageRef(
                mimetype="image/png",
                dpi=round(72 * image_scale),
                size=Size(width=float(table_image.width), height=float(table_image.height)),
                uri=from_pil_to_base64uri(table_image),
            )
            page_item = PageItem(
                page_no=page_index,
                size=Size(width=float(table_image.width), height=float(table_image.height)),
                image=image_ref,
            )

            html = "<table>" + "".join(item["html"]) + "</table>"
            table_data = convert_html_table_into_docling_tabledata(
                html, text_cells=item["cells"][0]
            )

            l = 0.0
            b = 0.0
            r = table_image.width
            t = table_image.height
            if "table_bbox" in item:
                l = item["table_bbox"][0]
                b = table_image.height - item["table_bbox"][3]
                r = item["table_bbox"][2]
                t = table_image.height - item["table_bbox"][1]

            bbox = BoundingBox(
                l=l,
                r=r,
                b=b,
                t=t,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )

            prov = ProvenanceItem(page_no=page_index, bbox=bbox, charspan=(0, 0))
            true_doc.pages[1] = page_item

            true_doc.add_table(data=table_data, caption=None, prov=prov)

            true_doc, _, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,  # pictures_column,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,  # page_images_column,
            )

            # In the dataset, item["image"] is a PIL Image. Convert it to bytes
            bytes_io = io.BytesIO()
            image = item["image"]
            image.save(bytes_io, format="png")
            image_bytes = bytes_io.getvalue()
            image_stream = DocumentStream(name=filename, stream=BytesIO(image_bytes))
            record = DatasetRecord(
                predictor_info=self.prediction_provider.info(),
                doc_id=str(filename),
                ground_truth_doc=true_doc,
                doc_hash=get_binhash(image_bytes),
                original=image_stream,
                mime_type="image/png",
            )

            # Create the prediction, convert it to doclingDocument and update the dataset record
            # Note: This saves the prediction and its doclingDocument as .json in the target directory
            self.update_prediction(record)

            # Save the ground truth data as well - for debugging
            output_dir = self.target / "microsoft" / "ground_truth_docling_document"
            os.makedirs(output_dir, exist_ok=True)
            docling_document_file_name = os.path.join(output_dir, f"{filename}.json")
            with open(docling_document_file_name, 'w', encoding="utf-8") as f:
                json.dump(true_doc.export_to_dict(), f, indent=2)

            # If visualization flag is set, run the visualizations and save them a well
            if self.do_visualization and record.predicted_doc is not None:
                save_comparison_html(
                    filename=viz_dir / f"{os.path.basename(filename)}.html",
                    true_doc=true_doc,
                    pred_doc=record.predicted_doc,
                    page_image=true_page_images[0],
                    true_labels=TRUE_HTML_EXPORT_LABELS,
                    pred_labels=PRED_HTML_EXPORT_LABELS,
                )

            yield record
