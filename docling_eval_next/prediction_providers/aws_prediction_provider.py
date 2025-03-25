import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from docling_core.types import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream
from PIL import Image

from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)


class AWSTextractPredictionProvider(BasePredictionProvider):
    """Provider that calls the AWS Textract API for predicting the tables in document."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id:
            raise ValueError(
                "Environment variable AWS_ACCESS_KEY_ID is not set. This is required to authenticate with AWS."
            )
        if not aws_secret_access_key:
            raise ValueError(
                "Environment variable AWS_SECRET_ACCESS_KEY is not set. This is required to authenticate with AWS."
            )

        region_name = os.getenv("AWS_REGION", "us-east-1")

        self.textract_client = boto3.client("textract", region_name=region_name)

        self.skip_api_if_prediction_is_present = bool(
            kwargs.get("skip_api_if_prediction_is_present", False) is True
        )
        self.predictions_dir = kwargs.get("predictions_dir")
        os.makedirs(self.predictions_dir, exist_ok=True)

    def extract_bbox_from_geometry(self, geometry):
        """Helper function to extract bbox coordinates from AWS Textract geometry data."""
        if not geometry or not geometry.get("BoundingBox"):
            return {"l": 0, "t": 0, "r": 0, "b": 0}

        bbox = geometry.get("BoundingBox")

        left = bbox.get("Left", 0)
        top = bbox.get("Top", 0)
        width = bbox.get("Width", 0)
        height = bbox.get("Height", 0)

        return {"l": left, "t": top, "r": left + width, "b": top + height}

    def get_cell_content(self, cell, blocks_map):
        """Extract text content from a table cell."""
        text = ""
        if "Relationships" in cell:
            for relationship in cell.get("Relationships", []):
                if relationship["Type"] == "CHILD":
                    for child_id in relationship["Ids"]:
                        child = blocks_map[child_id]
                        if child["BlockType"] == "WORD":
                            text += child["Text"] + " "
        return text.strip()

    def process_table(self, table, blocks_map, page_no):
        """Process a single table from AWS Textract output."""
        table_cells = []

        table_bbox = self.extract_bbox_from_geometry(table.get("Geometry", {}))
        table_bbox_obj = BoundingBox(
            l=table_bbox["l"],
            t=table_bbox["t"],
            r=table_bbox["r"],
            b=table_bbox["b"],
            coord_origin=CoordOrigin.TOPLEFT,
        )

        table_prov = ProvenanceItem(
            page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0)
        )

        cells = []
        if "Relationships" in table:
            for relationship in table.get("Relationships", []):
                if relationship["Type"] == "CHILD":
                    for cell_id in relationship["Ids"]:
                        cell = blocks_map[cell_id]
                        if cell["BlockType"] == "CELL":
                            cells.append(cell)

        max_row = 0
        max_col = 0
        for cell in cells:
            row = cell.get("RowIndex", 0)
            col = cell.get("ColumnIndex", 0)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        for cell in cells:
            row_index = cell.get("RowIndex", 1) - 1
            col_index = cell.get("ColumnIndex", 1) - 1
            row_span = cell.get("RowSpan", 1)
            col_span = cell.get("ColumnSpan", 1)

            # Get cell content
            cell_text = self.get_cell_content(cell, blocks_map)

            # Get cell bbox
            cell_bbox = self.extract_bbox_from_geometry(cell.get("Geometry", {}))

            # Create TableCell object
            table_cell = TableCell(
                bbox=BoundingBox(
                    l=cell_bbox["l"],
                    t=cell_bbox["t"],
                    r=cell_bbox["r"],
                    b=cell_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_index,
                end_row_offset_idx=row_index + row_span,
                start_col_offset_idx=col_index,
                end_col_offset_idx=col_index + col_span,
                text=cell_text,
                # AWS doesn't directly tell us which cells are headers, so we're using heuristics
                # Setting first row as column header and first column as row header
                column_header=(row_index == 0),
                row_header=(col_index == 0),
                row_section=False,
            )

            table_cells.append(table_cell)

        table_data = TableData(
            table_cells=table_cells, num_rows=max_row, num_cols=max_col
        )

        return table_prov, table_data

    def convert_aws_output_to_docling(
        self, analyze_result, filename, file_bytes
    ) -> DoclingDocument:
        """Converts AWS Textract output to DoclingDocument format."""
        doc_id = Path(filename).stem
        doc = DoclingDocument(name=filename)

        blocks_map = {block["Id"]: block for block in analyze_result.get("Blocks", [])}

        processed_pages = set()

        # Get page dimensions from page block
        # AWS provides normalized coordinates, so we need to multiply by a typical page size
        # width = 8.5 * 72  # Standard US Letter width in points
        # height = 11 * 72  # Standard US Letter height in points
        im = Image.open(BytesIO(file_bytes))
        width, height = im.size

        for block in analyze_result.get("Blocks", []):
            if block["BlockType"] == "PAGE":
                page_no = int(block.get("Page", 1))
                processed_pages.add(page_no)

                doc.pages[page_no] = PageItem(
                    size=Size(width=float(width), height=float(height)),
                    page_no=page_no,
                )

            if block["BlockType"] == "WORD" and block.get("Page", 1) == page_no:
                text_content = block.get("Text", "")
                bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))

                # Scale normalized coordinates to the page dimensions
                bbox_obj = BoundingBox(
                    l=bbox["l"] * width,
                    t=bbox["t"] * height,
                    r=bbox["r"] * width,
                    b=bbox["b"] * height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                prov = ProvenanceItem(
                    page_no=page_no,
                    bbox=bbox_obj,
                    charspan=(0, len(text_content)),
                )

                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

            if block["BlockType"] == "TABLE":
                page_no = int(block.get("Page", 1))
                table_prov, table_data = self.process_table(block, blocks_map, page_no)
                doc.add_table(prov=table_prov, data=table_data, caption=None)

        return doc

    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        **extra_args,
    ) -> DoclingDocument:
        """For the given document stream (single document), run the API and create the doclingDocument."""

        if stream is None or stream.name is None:
            raise ValueError("The 'stream' parameter or its 'name' attribute is None.")

        print(f"Creating prediction for file - {stream.name}..")
        stream_file_basename = Path(stream.name).stem

        raw_prediction_file_name = os.path.join(
            self.predictions_dir, f"{gt_doc.name}.raw.json"
        )
        docling_document_file_name = os.path.join(
            self.predictions_dir, f"{gt_doc.name}.docling.json"
        )

        # Check if the prediction exists, if so - reuse it
        if self.skip_api_if_prediction_is_present and os.path.exists(
            raw_prediction_file_name
        ):
            print(
                f"Skipping AWS API call and re-using existing prediction from [{raw_prediction_file_name}]."
            )
            with open(raw_prediction_file_name, "r", encoding="utf-8") as f:
                result_json = json.load(f)
        else:

            file_bytes = stream.stream.read()
            stream.stream.seek(0)
            response = self.textract_client.analyze_document(
                Document={"Bytes": file_bytes}, FeatureTypes=["TABLES", "FORMS"]
            )

            try:
                with open(raw_prediction_file_name, "w") as outfile:
                    json.dump(response, outfile, indent=4)
                print(f"JSON response saved to '{raw_prediction_file_name}'")
            except IOError as e:
                print(f"Error saving JSON to file: {e}")

            result_json = json.loads(json.dumps(response, default=str))
            print(f"Successfully processed [{stream.name}] using AWS Textract API!")

        pred_docling_doc = self.convert_aws_output_to_docling(
            result_json, raw_prediction_file_name, stream.stream.read()
        )

        with open(docling_document_file_name, "w", encoding="utf-8") as f:
            json.dump(pred_docling_doc.export_to_dict(), f, indent=2)
        print(
            f"Saved Docling Document output of prediction to - {docling_document_file_name}"
        )

        return pred_docling_doc

    def info(self) -> Dict:
        return {"asset": "AWS Textract", "version": "1.0.0"}
