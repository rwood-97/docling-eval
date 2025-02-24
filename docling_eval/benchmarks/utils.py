import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup  # type: ignore
from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    TableCell,
    TableData,
)
from PIL import Image

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.converters.utils import from_pil_to_base64uri


def get_binhash(binary_data: bytes) -> str:
    # Create a hash object (e.g., SHA-256)
    hash_object = hashlib.sha256()

    # Update the hash object with the binary data
    hash_object.update(binary_data)

    # Get the hexadecimal digest of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def write_datasets_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
):
    features = Features(
        {
            BenchMarkColumns.DOCLING_VERSION: Value("string"),
            BenchMarkColumns.STATUS: Value("string"),
            BenchMarkColumns.DOC_ID: Value("string"),
            BenchMarkColumns.DOC_PATH: Value("string"),
            BenchMarkColumns.DOC_HASH: Value("string"),
            BenchMarkColumns.GROUNDTRUTH: Value("string"),
            BenchMarkColumns.GROUNDTRUTH_PICTURES: Sequence(Features_Image()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: Sequence(Features_Image()),
            BenchMarkColumns.PREDICTION: Value("string"),
            BenchMarkColumns.PREDICTION_PICTURES: Sequence(Features_Image()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: Sequence(Features_Image()),
            BenchMarkColumns.ORIGINAL: Value("string"),
            BenchMarkColumns.MIMETYPE: Value("string"),
            BenchMarkColumns.MODALITIES: Sequence(Value("string")),
        }
    )

    schema = features.to_dict()
    # print(json.dumps(schema, indent=2))

    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": schema,
            "num_rows": num_train_rows,
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": schema,
            "num_rows": num_test_rows,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_infos.json", "w") as fw:
        json.dump(dataset_infos, fw, indent=2)


def get_input_document(file: Path | BytesIO) -> InputDocument:
    return InputDocument(
        path_or_stream=file,
        format=InputFormat.PDF,  # type: ignore[arg-type]
        filename=file.name if isinstance(file, Path) else "foo",
        backend=DoclingParseV2DocumentBackend,
    )


def add_pages_to_true_doc(
    pdf_path: Path | BytesIO, true_doc: DoclingDocument, image_scale: float = 1.0
):
    in_doc = get_input_document(pdf_path)
    assert in_doc.valid, "Input doc must be valid."
    # assert in_doc.page_count == 1, "doc must have one page."

    # add the pages
    page_images: List[Image.Image] = []

    for page_no in range(0, in_doc.page_count):
        page = Page(page_no=page_no)
        page._backend = in_doc._backend.load_page(page.page_no)  # type: ignore[attr-defined]

        if page._backend is not None and page._backend.is_valid():
            page.size = page._backend.get_size()

            page_width, page_height = page.size.width, page.size.height

            page_image = page.get_image(scale=image_scale)
            if page_image is not None:
                page_images.append(page_image)
                image_ref = ImageRef(
                    mimetype="image/png",
                    dpi=round(72 * image_scale),
                    size=Size(
                        width=float(page_image.width), height=float(page_image.height)
                    ),
                    # uri=Path(f"{BenchMarkColumns.PAGE_IMAGES}/{page_no}"),
                    uri=from_pil_to_base64uri(page_image),
                )
                page_item = PageItem(
                    page_no=page_no + 1,
                    size=Size(width=float(page_width), height=float(page_height)),
                    image=image_ref,
                )

                true_doc.pages[page_no + 1] = page_item
                # page_image.show()
            else:
                logging.warning("did not get image for page `add_pages_to_true_doc`")

            page._backend.unload()

    return true_doc, page_images


def yield_cells_from_html_table(
    table_html: str, text_cells: Optional[List[Dict]] = None
):
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup  # Ensure table context
    rows = table.find_all("tr")

    max_cols = 0
    for row in rows:
        # cols = row.find_all(["td", "th"])
        # max_cols = max(max_cols, len(cols))  # Determine maximum columns

        num_cols = 0
        for cell in row.find_all(["td", "th"]):
            num_cols += int(cell.get("colspan", 1))

        max_cols = max(max_cols, num_cols)  # Determine maximum columns

    # Create grid to track cell positions
    grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

    text_cell_id = 0
    for row_idx, row in enumerate(rows):
        col_idx = 0  # Start from first column
        for cell in row.find_all(["td", "th"]):
            # Skip over filled grid positions (handle previous rowspan/colspan)
            while grid[row_idx][col_idx] is not None:
                col_idx += 1

            # Get text, rowspan, and colspan
            text = cell.get_text(strip=True)

            if len(text) == 0 and text_cells is not None:
                text_cell = text_cells[text_cell_id]
                text = "".join(text_cell["tokens"])

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill grid positions and yield (row, column, text)
            for r in range(rowspan):
                for c in range(colspan):
                    grid[row_idx + r][col_idx + c] = text

            # print(f"Row: {row_idx + 1}, Col: {col_idx + 1}, Text: {text}")
            yield row_idx, col_idx, rowspan, colspan, text

            col_idx += colspan  # Move to next column after colspan

            text_cell_id += 1


def convert_html_table_into_docling_tabledata(
    table_html: str, text_cells: Optional[List] = None
) -> TableData:

    num_rows = -1
    num_cols = -1

    cells = []

    try:
        for row_idx, col_idx, rowspan, colspan, text in yield_cells_from_html_table(
            table_html=table_html, text_cells=text_cells
        ):
            cell = TableCell(
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                text=text,
            )
            cells.append(cell)

            num_rows = max(row_idx + rowspan, num_rows)
            num_cols = max(col_idx + colspan, num_cols)

    except:
        logging.error("No table-structure identified")

    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
