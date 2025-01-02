import json
import logging
import os
import subprocess
from pathlib import Path
import glob
import copy

from tqdm import tqdm

from docling_eval.docling.conversion import create_converter

from docling_eval.docling.constants import (
    HTML_INSPECTION,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    ImageRefMode,    
    PageItem,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)

from docling_eval.benchmarks.utils import (
    draw_clusters_with_reading_order
)

from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64,
    from_pil_to_base64uri,
    get_binary,
    save_shard_to_disk,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

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

def parse_args():

    name = "package-forms"
    pdf_files = sorted(glob.glob("/Users/taa/Documents/projects/docling-scrape/downloaded_files/*.pdf"))
    return name, pdf_files, 1.0

def main():
    
    name, pdf_files, image_scale = parse_args()

    odir = Path(f"./benchmarks/{name}")

    pqt_dir = odir / "dataset"
    viz_dir = odir / "visualization"

    os.makedirs(pqt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create Converter
    doc_converter = create_converter(
        artifacts_path=odir / "artifacts", page_image_scale=image_scale
    )
    
    for pdf_file in tqdm(pdf_files, total=len(pdf_files), ncols=128):

        try:
            # Create the predicted Document
            conv_results = doc_converter.convert(source=pdf_file, raises_on_error=True)
            pred_doc = conv_results.document
        except:
            continue
            
        html_doc = pred_doc.export_to_html(
            image_mode=ImageRefMode.EMBEDDED,
            #html_head=HTML_DEFAULT_HEAD_FOR_COMP,
            #labels=pred_labels,
        )

        html_doc = html_doc.replace("'", "&#39;")
        
        page_images = []
        page_template = '<div class="image-wrapper"><img src="data:image/png;base64,BASE64PAGE" alt="Example Image"></div>' 
        for page_no, page in pred_doc.pages.items():
            page_img = page.image.pil_image

            page_img = draw_clusters_with_reading_order(doc=pred_doc,
                                                        page_image=page_img,
                                                        labels=PRED_HTML_EXPORT_LABELS,
                                                        page_no=page_no,
                                                        reading_order=True)
            
            page_base64 = from_pil_to_base64(page_img)
            page_images.append(page_template.replace("BASE64PAGE", page_base64))

        page = copy.deepcopy(HTML_INSPECTION)
        page = page.replace("PREDDOC", html_doc)
        page = page.replace("PAGE_IMAGES", "\n".join(page_images))

        filename = viz_dir / f"{os.path.basename(pdf_file)}.html"
        logging.info(f"writing {filename}")
        with open(str(filename), "w") as fw:
            fw.write(page)
    
if __name__ == "__main__":
    main()
