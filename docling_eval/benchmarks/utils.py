import copy
import json
from pathlib import Path
import io
import base64
from PIL import Image as PILImage
from typing import Dict, List
import pypdfium2 as pdfium

from bs4 import BeautifulSoup  # type: ignore

from docling_core.types.doc.labels import DocItemLabel

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size

from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem, ImageRefMode
)

from docling_eval.benchmarks.constants import BenchMarkColumns, BenchMarkNames

from docling_eval.docling.constants import HTML_DEFAULT_HEAD_FOR_COMP, HTML_COMPARISON_PAGE


def write_datasets_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
):

    columns = [
        {"name": BenchMarkColumns.DOCLING_VERSION, "type": "string"},
        {"name": BenchMarkColumns.STATUS, "type": "string"},
        {"name": BenchMarkColumns.DOC_ID, "type": "string"},
        {"name": BenchMarkColumns.GROUNDTRUTH, "type": "string"},
        {"name": BenchMarkColumns.PREDICTION, "type": "string"},
        {"name": BenchMarkColumns.ORIGINAL, "type": "string"},
        {"name": BenchMarkColumns.MIMETYPE, "type": "string"},
        {"name": BenchMarkColumns.PICTURES, "type": {"list": {"item": "Image"}}},
        {"name": BenchMarkColumns.PAGE_IMAGES, "type": {"list": {"item": "Image"}}},
    ]

    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": {"columns": columns},
            "num_rows": num_train_rows,
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": {"columns": columns},
            "num_rows": num_test_rows,
        },
    }

    with open(output_dir / f"dataset_infos.json", "w") as fw:
        fw.write(json.dumps(dataset_infos, indent=2))

def add_pages_to_true_doc(pdf_path:Path, true_doc:DoclingDocument, image_scale:float = 1.0):

    pdf = pdfium.PdfDocument(pdf_path)
    assert len(pdf) == 1, "len(pdf)==1"

    # add the pages
    page_images: List[PILImage.Image] = []

    pdf = pdfium.PdfDocument(pdf_path)
    for page_index in range(len(pdf)):
        # Get the page
        page = pdf.get_page(page_index)

        # Get page dimensions
        page_width, page_height = page.get_width(), page.get_height()

        # Render the page to an image
        page_image = page.render(scale=image_scale).to_pil()

        page_images.append(page_image)

        # Close the page to free resources
        page.close()

        image_ref = ImageRef(
            mimetype="image/png",
            dpi=round(72 * image_scale),
            size=Size(
                width=float(page_image.width), height=float(page_image.height)
            ),
            uri=Path(f"{BenchMarkColumns.PAGE_IMAGES}/{page_index}"),
        )
        page_item = PageItem(
            page_no=page_index + 1,
            size=Size(width=float(page_width), height=float(page_height)),
            image=image_ref,
        )

        true_doc.pages[page_index + 1] = page_item
    
    return true_doc, page_images

def yield_cells_from_html_table(table_html:str):
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup  # Ensure table context
    rows = table.find_all("tr")

    max_cols = 0
    for row in rows:
        # cols = row.find_all(["td", "th"])
        # max_cols = max(max_cols, len(cols))  # Determine maximum columns

        num_cols=0
        for cell in row.find_all(["td", "th"]):
            num_cols += int(cell.get("colspan", 1))

        max_cols = max(max_cols, num_cols)  # Determine maximum columns

    # Create grid to track cell positions
    grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

    for row_idx, row in enumerate(rows):
        col_idx = 0  # Start from first column
        for cell in row.find_all(["td", "th"]):
            # Skip over filled grid positions (handle previous rowspan/colspan)
            while grid[row_idx][col_idx] is not None:
                col_idx += 1

            # Get text, rowspan, and colspan
            text = cell.get_text(strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill grid positions and yield (row, column, text)
            for r in range(rowspan):
                for c in range(colspan):
                    grid[row_idx + r][col_idx + c] = text

            # print(f"Row: {row_idx + 1}, Col: {col_idx + 1}, Text: {text}")
            yield row_idx, col_idx, rowspan, colspan, text

            col_idx += colspan  # Move to next column after colspan

def convert_html_table_into_docling_tabledata(table_html:str) -> TableData:

    num_rows = -1
    num_cols = -1

    cells = []
            
    try:
        for row_idx, col_idx, rowspan, colspan, text in yield_cells_from_html_table(
                table_html=table_html
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

            
def save_comparison_html(filename:Path, true_doc:DoclingDocument, pred_doc:DoclingDocument, page_image:PILImage, labels: List[DocItemLabel]):

    true_doc_html = true_doc.export_to_html(image_mode = ImageRefMode.EMBEDDED,
                                            html_head = HTML_DEFAULT_HEAD_FOR_COMP, labels=labels)
    
    pred_doc_html = pred_doc.export_to_html(image_mode = ImageRefMode.EMBEDDED,
                                            html_head = HTML_DEFAULT_HEAD_FOR_COMP, labels=labels)
    
    # since the string in srcdoc are wrapped by ', we need to replace all ' by it HTML convention
    true_doc_html = true_doc_html.replace("'", "&#39;")
    pred_doc_html = pred_doc_html.replace("'", "&#39;")
            
    # Convert the image to a bytes object
    buffered = io.BytesIO()
    page_image.save(buffered, format="PNG")  # Specify the format (e.g., JPEG, PNG, etc.)
    image_bytes = buffered.getvalue()
    
    # Encode the bytes to a Base64 string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    comparison_page = copy.deepcopy(HTML_COMPARISON_PAGE)
    comparison_page = comparison_page.replace("BASE64PAGE", image_base64)
    comparison_page = comparison_page.replace("TRUEDOC", true_doc_html)
    comparison_page = comparison_page.replace("PREDDOC", pred_doc_html)
    
    with open(str(filename), "w") as fw:
        fw.write(comparison_page)



