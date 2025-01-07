import argparse
import glob
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import xmltodict

from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.docling.utils import insert_images

from datasets import Dataset, load_dataset
from pathlib import Path
from PIL import Image  # as PILImage

from docling_core.types.doc.labels import DocItemLabel

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    ImageRef,
    PageItem,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)

from docling_eval.docling.utils import from_pil_to_base64, from_pil_to_base64uri
from docling_eval.benchmarks.utils import draw_clusters_with_reading_order


def rgb_to_hex(r, g, b):
    """
    Converts RGB values to a HEX color code.
    
    Args:
        r (int): Red value (0-255)
        g (int): Green value (0-255)
        b (int): Blue value (0-255)
        
    Returns:
        str: HEX color code (e.g., "#RRGGBB")
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in the range 0-255")
    
    return f"#{r:02X}{g:02X}{b:02X}"

def label_to_rgba_color(label:DocItemLabel):

    if label==DocItemLabel.TEXT: return (255, 255, 153)  # Light Yellow
    elif label==DocItemLabel.PARAGRAPH: return (255, 255, 153)  # Light Yellow
    elif label==DocItemLabel.REFERENCE: return (173, 216, 230)  # Light Yellow
    elif label==DocItemLabel.CAPTION: return (255, 204, 153)  # Light Orange
    elif label==DocItemLabel.LIST_ITEM: return (153, 153, 255)  # Light Purple
    elif label==DocItemLabel.FORMULA: return (192, 192, 192)  # Gray
    elif label==DocItemLabel.TABLE: return (255, 204, 204)  # Light Pink
    elif label==DocItemLabel.PICTURE: return (255, 204, 164)  # Light Beige
    elif label==DocItemLabel.SECTION_HEADER: return (255, 153, 153)  # Light Red
    elif label==DocItemLabel.PAGE_HEADER: return (204, 255, 204)  # Light Green
    elif label==DocItemLabel.PAGE_FOOTER: return (204, 255, 204)  # Light Green 
    elif label==DocItemLabel.TITLE: return (255, 153, 153)  # Light Red (same as Section-Header)
    elif label==DocItemLabel.FOOTNOTE: return (200, 200, 255)  # Light Blue
    elif label==DocItemLabel.DOCUMENT_INDEX: return (220, 220, 220)  # Light Gray
    elif label==DocItemLabel.CODE: return (125, 125, 125)  # Gray
    elif label==DocItemLabel.CHECKBOX_SELECTED: return (255, 182, 193)  # Pale Green
    elif label==DocItemLabel.CHECKBOX_UNSELECTED: return (255, 182, 193)  # Light Pink
    elif label==DocItemLabel.FORM: return (200, 255, 255)  # Light Cyan
    elif label==DocItemLabel.KEY_VALUE_REGION: return (183, 65, 14)  # Rusty orange
    elif label==DocItemLabel.FORM: return (64, 64, 64)
    elif label==DocItemLabel.KEY_VALUE_REGION: return (255, 255, 0)
    elif label==DocItemLabel.PICTURE: return (150, 123, 182)
    else:
        logger.warning(f"no color for label: {label}")

def label_to_hex_color(label:DocItemLabel):
    
    r, g, b = label_to_rgba_color(label)
    return rgb_to_hex(r, g, b)

def line_to_rgba_color(line:str):

    if line=="reading_order":
        return (255, 0, 0)
    elif line=="next_text":
        return (255, 0, 255)
    elif line=="to_caption":
        return (0, 255, 0)
    elif line=="to_footnote":
        return (0, 255, 0)
    elif line=="to_value":
        return (0, 0, 255)            
        
def line_to_hex_color(line:str):

    r, g, b = line_to_rgba_color(line)
    return rgb_to_hex(r, g, b)

def find_box(boxes: List, point: Tuple[float, float]):

    index=-1
    area = 1e6
    
    for i,box in enumerate(boxes):
        assert box["l"]<box["r"]
        assert box["b"]>box["t"]
        
        if box["l"]<=point[0] and point[0]<=box["r"] and \
           box["t"]<=point[1] and point[1]<=box["b"]:
            if abs(box["r"]-box["l"])*(box["b"]-box["t"])<area:
                area = abs(box["r"]-box["l"])*(box["b"]-box["t"])
                index = i

    assert index!=-1
                
    return index, boxes[index]

def create_prov(box:Dict, page_no:int):
    
    bbox = BoundingBox(
        l=box["l"],
        r=box["r"],
        b=box["b"],
        t=box["t"],
        coord_origin=CoordOrigin.TOPLEFT,
    )

    prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))

    return prov
    
def from_cvat_to_docling_document(annotation_filenames:List[Path], page_images_dir:Path, image_scale:float=1.0):

    for annot_file in annotation_filenames:

        with open(str(annot_file), "r") as fr:
            xml_data = fr.read()

        # Convert XML to a Python dictionary
        annot_data = xmltodict.parse(xml_data)        

        print(annot_data)
        for image_annot in annot_data["annotations"]["image"]:
            name = image_annot["@name"]
            
            print("image: ", image_annot["@name"])
        
            boxes = []
            if isinstance(image_annot["box"], dict):
                boxes = [image_annot["box"]]
            if isinstance(image_annot["box"], list):
                boxes = image_annot["box"]
                
            lines = []
            if isinstance(image_annot["polyline"], dict):
                lines = [image_annot["polyline"]]
            if isinstance(image_annot["polyline"], list):
                lines = image_annot["polyline"]

            for i,box in enumerate(boxes):
                boxes[i]["b"] = float(box["@ybr"])
                boxes[i]["t"] = float(box["@ytl"])
                boxes[i]["l"] = float(box["@xtl"])
                boxes[i]["r"] = float(box["@xbr"])

                print("\t", boxes[i])
                assert boxes[i]["b"]>boxes[i]["t"]
                
            for i,line in enumerate(lines):
                print("\t", line)

                points=[]
                for _ in line["@points"].split(";"):
                    __ = _.split(",")
                    points.append((float(__[0]), float(__[1])))
                    
                lines[i]["points"] = points

            page_image = Image.open(str(page_images_dir / name))
                
            true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(name)}")
            
            page_index = 1

            image_ref = ImageRef(
                mimetype="image/png",
                dpi=round(72 * image_scale),
                size=Size(width=float(page_image.width), height=float(page_image.height)),
                uri=from_pil_to_base64uri(page_image),
            )
            page_item = PageItem(
                page_no=page_index,
                size=Size(width=float(page_image.width), height=float(page_image.height)),
                image=image_ref,
            )
            true_doc.pages[page_index] = page_item
            
            for i,line in enumerate(lines):
                if line["@label"]=="reading_order":
                    for point in line["points"]:
                        ind, box = find_box(boxes=boxes, point=point)
                        # print(ind, ": ", box)
                        
                        prov = create_prov(box=box, page_no=page_index)
                        print(" => ", prov)
                        
                        label = box["@label"]

                        if label==DocItemLabel.TEXT:
                            true_doc.add_text(label=label, prov=prov, text="")
                        elif label==DocItemLabel.PARAGRAPH:
                            true_doc.add_text(label=label, prov=prov, text="")
                        elif label==DocItemLabel.REFERENCE:
                            true_doc.add_text(label=label, prov=prov, text="")
                        elif label==DocItemLabel.CAPTION:
                            pass
                        elif label==DocItemLabel.LIST_ITEM:
                            true_doc.add_listitem(prov=prov, text="")
                        elif label==DocItemLabel.FORMULA:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.TABLE:
                            true_doc.add_table(prov=prov)                            
                        elif label==DocItemLabel.PICTURE:
                            true_doc.add_picture(prov=prov)                            
                        elif label==DocItemLabel.SECTION_HEADER:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.PAGE_HEADER:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.PAGE_FOOTER:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.TITLE:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.FOOTNOTE:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.DOCUMENT_INDEX:
                            true_doc.add_table(label=DocItemLabel.DOCUMENT_INDEX, prov=prov)                            
                        elif label==DocItemLabel.CODE:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.CHECKBOX_SELECTED:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.CHECKBOX_UNSELECTED:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.FORM:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.KEY_VALUE_REGION:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.FORM:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        elif label==DocItemLabel.KEY_VALUE_REGION:
                            true_doc.add_text(label=label, prov=prov, text="")                            
                        else:
                            logging.error(f"unknown label={label}")

            """
            md = true_doc.export_to_markdown()
            print(md)
            """
            
            yield true_doc, [page_image]
            
def from_docling_document_to_cvat(docs: List[DoclingDocument],
                                  output_dir:Path):

    output_page_dir = output_dir / "page_images"

    for _ in [output_dir, output_page_dir]:
        os.makedirs(_, exist_ok=True)

    results=[]

    results.append('<?xml version="1.0" encoding="utf-8"?>')
    results.append('<annotations>')
    results.append('<version>1.1</version>')
    results.append('<meta>')
    results.append('<task>')
    
    results.append("<labels>")
    for item in DocItemLabel:
        results.append("<label>")
        results.append(f"<name>{item.value}</name>")
        results.append(f"<color>{label_to_hex_color(item)}</color>")
        results.append(f"<type>rectangle</type>")
        results.append(f"<attributes></attributes>")
        results.append("</label>")
        
    for value in ["reading_order", "next_text", "to_caption", "to_footnote", "to_value"]:
        results.append("<label>")
        results.append(f"<name>{value}</name>")
        results.append(f"<color>{line_to_hex_color(value)}</color>")
        results.append(f"<type>polyline</type>")
        results.append(f"<attributes></attributes>")
        results.append("</label>")
        
    results.append("</labels>")
    results.append('</task>')    
    results.append('</meta>')
    
    img_id = 0
    for doc_id,doc in enumerate(docs):

        page_images = []
        page_fnames = []
        for j,page in doc.pages.items():
            page_image = page.image.pil_image
            page_image.save(str(output_page_dir / f"doc_{doc_id:06}_{j:06}.png"))
            
            page_images.append(page_image)
            page_fnames.append(f"doc_{doc_id:06}_{j:06}.png")

        page_bboxes = {i: [] for i,fname in enumerate(page_fnames)}
        for item, level in doc.iterate_items():
            if isinstance(item, DocItem): # and item.label in labels:
                for prov in item.prov:
                    page_no = prov.page_no

                    page_w = doc.pages[prov.page_no].size.width
                    page_h = doc.pages[prov.page_no].size.height

                    img_w = page_images[page_no-1].width
                    img_h = page_images[page_no-1].height
                    
                    page_bbox = prov.bbox.to_top_left_origin(page_height=page_h)
                    
                    img_bbox = [
                        page_bbox.l/page_w*img_w,
                        page_bbox.b/page_h*img_h,
                        page_bbox.r/page_w*img_w,
                        page_bbox.t/page_h*img_h,
                    ]

                    page_bboxes[page_no-1].append({
                        "label": item.label.value,
                        "l": img_bbox[0],
                        "r": img_bbox[2],
                        "b": img_bbox[1],
                        "t": img_bbox[3]
                    })
        
        for page_no, page_file in enumerate(page_fnames):
            img_w = page_images[page_no].width
            img_h = page_images[page_no].height
            
            results.append(f'<image id="{img_id}" name="{page_file}" width="{img_w}" height="{img_h}">')

            for bbox_id, page_bbox in enumerate(page_bboxes[page_no]):
                print(page_bbox)
                label = page_bbox["label"]
                l = round(page_bbox["l"])
                r = round(page_bbox["r"])
                t = round(page_bbox["t"])
                b = round(page_bbox["b"])                
                results.append(f'<box label="{label}" source="manual" occluded="0" xtl="{l}" ytl="{t}" xbr="{r}" ybr="{b}" z_order="{bbox_id}"></box>')
            
            results.append("</image>")

    results.append('</annotations>')

    with open(str(output_dir / "pre-annotations.xml"), "w") as fw:
        fw.write("\n".join(results))

    xml_data = "\n".join(results)
    # Convert XML to a Python dictionary
    dict_data = xmltodict.parse(xml_data)

    # Convert the dictionary to a JSON string
    json_data = json.dumps(dict_data, indent=4)    
    #print(json_data)
        

def main_0():

    benchmark_path = Path("./benchmarks/DPBench-dataset/layout/test")
    
    test_files = sorted(glob.glob(str(benchmark_path / "*.parquet")))
    ds = load_dataset("parquet", data_files={"test": test_files})

    logging.info(f"oveview of dataset: {ds}")

    if ds is not None:
        ds_selection = ds["test"]

    docs = []
    for i, data in tqdm(
        enumerate(ds_selection),
        desc="iterating dataset",
        ncols=120,
        total=len(ds_selection),
    ):
        
        pred_doc_dict = data[BenchMarkColumns.PREDICTION]
        pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

        page_images = data[BenchMarkColumns.PAGE_IMAGES]
        pics_images = data[BenchMarkColumns.PICTURES]

        print("#-pics-images: ", len(pics_images))
        print("#-page-images: ", len(page_images))
        
        insert_images(pred_doc,
                      page_images=data[BenchMarkColumns.PAGE_IMAGES],
                      pictures=data[BenchMarkColumns.PICTURES])
        
        docs.append(pred_doc)

        if i>3:
            break

    from_docling_document_to_cvat(docs=docs, output_dir = Path("./benchmarks/preannotations"))

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
    
def main():

    annots = [Path("./benchmarks/preannotations/annotations.xml")]
    page_images = Path("./benchmarks/preannotations/page_images")

    for true_doc, true_page_images in from_cvat_to_docling_document(annots, page_images):

        """
        md = true_doc.export_to_markdown()
        print("markdown: \n\n", md)
        """

        for item, level in true_doc.iterate_items():
            if isinstance(item, DocItem): # and item.label in labels:
                for prov in item.prov:
                    print(item.label, " => ", prov.bbox)

        
        true_page_images[0].show()
        print("start drawing ...")
        res = draw_clusters_with_reading_order(doc=true_doc,
                                               page_image=true_page_images[0],
                                               labels=TRUE_HTML_EXPORT_LABELS)
        res.show()
        
if __name__ == "__main__":
    main()        
    
    
