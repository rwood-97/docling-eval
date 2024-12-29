import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import glob

import pypdfium2 as pdfium
from tqdm import tqdm  # type: ignore

from bs4 import BeautifulSoup  # type: ignore

from docling_core.types.doc.labels import DocItemLabel

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

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import write_datasets_info
from docling_eval.docling.conversion import create_converter

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size

from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    get_binary,
    save_shard_to_disk,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_filenames(omnidocbench_dir: Path):

    page_images = sorted(glob.glob(str(omnidocbench_dir / "images/*.jpg")))
    page_pdfs = sorted(glob.glob(str(omnidocbench_dir / "ori_pdfs/*.pdf")))

    assert len(page_images)==len(page_pdfs), f"len(page_images)!=len(page_pdfs) => {len(page_images)}!={len(page_pdfs)}"

    return list(zip(page_images, page_pdfs))
    

def update_gt_into_map(gt):

    result = {}
    
    for item in gt:
        path = item["page_info"]["image_path"]
        result[path] = item

    return result

def parse_html_table(table_html):
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup  # Ensure table context
    rows = table.find_all("tr")

    max_cols = 0
    for row in rows:
        cols = row.find_all(["td", "th"])
        max_cols = max(max_cols, len(cols))  # Determine maximum columns

    # Create grid to track cell positions
    grid = [[None for _ in range(max_cols * 2)] for _ in range(len(rows) * 2)]

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

def update_doc_with_gt(gt, true_doc, page_width:float, page_height:float):   

    print(json.dumps(gt, indent=2))
    
    gt_width = float(gt["page_info"]["width"])
    gt_height = float(gt["page_info"]["height"])
    
    for item in gt["layout_dets"]:

        print(json.dumps(item, indent=2))
        
        label = item["category_type"]

        text = ""
        if "text" in item:
            text = item["text"]
            
        min_x = item["poly"][0]
        max_x = item["poly"][0]

        min_y = item["poly"][1]
        max_y = item["poly"][1]

        for i in range(0,4):
            min_x = min(min_x, item["poly"][2*i])
            max_x = max(max_x, item["poly"][2*i])

            min_y = min(min_y, item["poly"][2*i+1])
            max_y = max(max_y, item["poly"][2*i+1])

        bbox = BoundingBox(
            l=min_x * page_width / gt_width,
            r=max_x * page_width / gt_width,
            b=min_y * page_height / gt_height,
            t=max_y * page_height / gt_height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))
        
        if label=="title":
            true_doc.add_heading(text=text, orig=text, level=1, prov=prov)
            
        elif label=="text_block":
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label=="text_mask":
            pass

        elif label=="table":

            num_rows = -1
            num_cols = -1

            cells = []
            
            if "html" in item:

                for row_idx, col_idx, rowspan, colspan, text in parse_html_table(
                        table_html=item["html"]
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

            else:
                logging.error("No table-structure identified")
                
            table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
            true_doc.add_table(data=table_data, caption=None, prov=prov)
            
        elif label=="table_caption":
            pass

        elif label=="table_footnote":
            pass
        
        elif label=="table_mask":
            pass
        
        elif label=="figure":
            true_doc.add_picture(prov=prov)
        
        elif label=="figure_caption":
            pass

        elif label=="figure_footnote":
            pass

        elif label=="equation_isolated":
            true_doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

        elif label=="equation_caption":
            pass

        elif label=="code_txt":
            pass
        
        elif label=="abandon":
            pass

        elif label=="need_mask":
            pass                

        elif label=="header":
            true_doc.add_text(label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov)
        
        elif label=="footer":
            true_doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov)

        elif label=="reference":
            pass

        elif label=="page_footnote":
            pass

        elif label=="page_number":
            pass
        
        else:
            logging.error(f"label {label} is not assigned!")
    
    return true_doc

    
def create_omnidocbench_e2e_dataset(
    omnidocbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):

    # Create Converter
    doc_converter = create_converter(
        artifacts_path=output_dir / "artifacts", page_image_scale=image_scale
    )
    
    # load the groundtruth
    with open(omnidocbench_dir / f"OmniDocBench.json", "r") as fr:
        gt = json.load(fr)

    gt = update_gt_into_map(gt)
        
    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []
        
    page_tuples = get_filenames(omnidocbench_dir)

    cnt = 0
    
    for page_tuple in tqdm(page_tuples, total=len(page_tuples), ncols=128, desc="Processing files for OmniDocBench with end-to-end"):

        jpg_path = page_tuple[0]
        pdf_path = page_tuple[1]
        
        logging.info(f"file: {pdf_path}")

        if not os.path.basename(jpg_path) in gt:
            logging.error(f"did not find ground-truth for {os.path.basename(jpg_path)}")
            continue

        gt_doc = gt[os.path.basename(jpg_path)]
        
        conv_results = doc_converter.convert(source=pdf_path, raises_on_error=True)

        conv_results.document.save_as_html(filename = viz_dir / f"{os.path.basename(pdf_path)}-pred.html",
                                           image_mode = ImageRefMode.EMBEDDED)

        pred_doc, pictures, page_images = extract_images(
            conv_results.document,
            pictures_column=BenchMarkColumns.PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.PAGE_IMAGES.value,  # page_images_column,
        )
        
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(jpg_path)}")
        true_doc.pages = pred_doc.pages

        page_width = pred_doc.pages[1].size.width
        page_height = pred_doc.pages[1].size.height

        true_doc = update_doc_with_gt(gt_doc, true_doc, page_width=page_width, page_height=page_height)

        true_doc.save_as_html(filename = viz_dir / f"{os.path.basename(pdf_path)}-true.html",
                              image_mode = ImageRefMode.PLACEHOLDER)
        
        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: "SUCCESS",
            BenchMarkColumns.DOC_ID: str(os.path.basename(jpg_path)),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.PAGE_IMAGES: page_images,
            BenchMarkColumns.PICTURES: pictures,
        }
        records.append(record)

        if cnt>=3:
            break
        else:
            cnt += 1
        
    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )
        
    
def create_omnidocbench_layout_dataset(
    omnidocbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    create_omnidocbench_e2e_dataset(
        omnidocbench_dir=omnidocbench_dir, output_dir=output_dir, image_scale=image_scale
    )


def create_omnidocbench_tableformer_dataset(
    omnidocbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    logger.error("Not implemented")
    exit(-1)


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Process DP-Bench benchmark from directory into HF dataset."
    )
    parser.add_argument(
        "-i",
        "--omnidocbench-directory",
        help="input directory with documents",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/omnidocbench",
    )
    parser.add_argument(
        "-s",
        "--image-scale",
        help="image-scale of the pages",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="mode of dataset",
        required=False,
        choices=["end-2-end", "table", "formula", "all"],
    )
    args = parser.parse_args()

    return (
        Path(args.omnidocbench_directory),
        Path(args.output_directory),
        float(args.image_scale),
        args.mode,
    )



def main():

    omnidocbench_dir, output_dir, image_scale, mode = parse_arguments()

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    odir_tab = Path(output_dir) / "tables"
    odir_eqn = Path(output_dir) / "formulas"

    os.makedirs(odir_e2e, exist_ok=True)
    os.makedirs(odir_tab, exist_ok=True)
    # os.makedirs(odir_eqn, exist_ok=True)

    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)
        os.makedirs(odir_tab / _, exist_ok=True)

    if mode == "end-2-end":
        create_omnidocbench_e2e_dataset(
            omnidocbench_dir=omnidocbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

    elif mode == "table":
        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=omnidocbench_dir, output_dir=odir_tab, image_scale=image_scale
        )

    elif mode == "all":
        create_omnidocbench_e2e_dataset(
            omnidocbench_dir=omnidocbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=omnidocbench_dir, output_dir=odir_tab, image_scale=image_scale
        )


if __name__ == "__main__":
    main()
