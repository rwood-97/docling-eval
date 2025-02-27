import io
import json
import logging
import math
import os
from pathlib import Path

from datasets import load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    GroupLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import (
    BenchMarkColumns,
    ConverterTypes,
    EvaluationModality,
)
from docling_eval.benchmarks.utils import (
    add_pages_to_true_doc,
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.converters.conversion import (
    create_pdf_docling_converter,
    create_smol_docling_converter,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


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

SHARD_SIZE = 100

category_map = {
    1: "caption",
    2: "footnote",
    3: "formula",
    4: "list_item",
    5: "page_footer",
    6: "page_header",
    7: "picture",
    8: "section_header",
    9: "table",
    10: "text",
    11: "title",
}


def ltwh_to_ltrb(box):
    l = box[0]
    t = box[1]
    w = box[2]
    h = box[3]
    r = l + w
    b = t + h
    return l, t, r, b


def update(true_doc, current_list, img, old_size, label, box, content):
    w, h = img.size
    new_size = Size(width=w, height=h)
    bbox = (
        BoundingBox.from_tuple(tuple(ltwh_to_ltrb(box)), CoordOrigin.TOPLEFT)
        .to_bottom_left_origin(page_height=old_size.height)
        .scale_to_size(old_size=old_size, new_size=new_size)
    )
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(content)))
    img_elem = crop_bounding_box(page_image=img, page=true_doc.pages[1], bbox=bbox)
    if label == DocItemLabel.PICTURE:
        current_list = None
        try:
            uri = from_pil_to_base64uri(img_elem)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img_elem.width, height=img_elem.height),
                uri=uri,
            )
        except Exception as e:
            log.error(
                "Warning: failed to resolve image uri for content {} of doc {}. Caught exception is {}:{}. Setting null ImageRef".format(
                    str(content), str(true_doc.name), type(e).__name__, e
                )
            )
            imgref = None

        true_doc.add_picture(prov=prov, image=imgref)
    elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
        current_list = None
        tbl_cell = TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=0,
            start_col_offset_idx=0,
            end_col_offset_idx=0,
            text=content,
        )
        tbl_data = TableData(table_cells=[tbl_cell])

        true_doc.add_table(data=tbl_data, prov=prov, label=label)
    elif label == DocItemLabel.LIST_ITEM:
        if current_list is None:
            current_list = true_doc.add_group(label=GroupLabel.LIST, name="list")

            # TODO: Infer if this is a numbered or a bullet list item
        true_doc.add_list_item(
            text=content, enumerated=False, prov=prov, parent=current_list
        )
    elif label == DocItemLabel.SECTION_HEADER:
        current_list = None
        true_doc.add_heading(text=content, prov=prov)
    else:
        current_list = None
        true_doc.add_text(label=label, text=content, prov=prov)


def create_dlnv1_e2e_dataset(
    name: str,
    split: str,
    output_dir: Path,
    converter_type: ConverterTypes = ConverterTypes.DOCLING,
    do_viz: bool = False,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    do_debug: bool = False,
):
    ds = load_dataset(name, split=split)
    total_ds_len = len(ds)

    # Check sample ranges
    if end_index == -1:
        end_index = total_ds_len

    if begin_index > end_index:
        raise RuntimeError("Cannot have from_sample_index > to_sample_index")

    if begin_index >= total_ds_len or end_index > total_ds_len:
        raise IndexError(
            f"The sample indices go beyond the dataset size: {total_ds_len}"
        )

    # Select the asked rows
    ds = ds.select(range(begin_index, end_index))
    selected_ds_len = len(ds)

    log.info(
        "Dataset len: %s. Selected range: [%s, %s] = %s",
        total_ds_len,
        begin_index,
        end_index,
        selected_ds_len,
    )

    # Decide which converter type to initialize
    if converter_type == ConverterTypes.DOCLING:
        converter = create_pdf_docling_converter(page_image_scale=1.0)
    else:
        converter = create_smol_docling_converter()

    if do_viz:
        viz_dir = output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)

    test_dir = output_dir / split
    os.makedirs(test_dir, exist_ok=True)
    records = []
    exported_rows = 0
    skipped_rows = 0
    saved_shards = 0

    for doc in tqdm(ds, total=selected_ds_len):
        try:
            page_hash = doc["metadata"]["page_hash"]

            if do_debug:
                log.debug("Converting: %s", page_hash)

            pdf = doc["pdf"]
            pdf_stream = io.BytesIO(pdf)
            pdf_stream.seek(0)
            conv_results = converter.convert(
                source=DocumentStream(
                    name=doc["metadata"]["page_hash"], stream=pdf_stream
                ),
                raises_on_error=True,
            )
            pdf_stream = io.BytesIO(pdf)

            pred_doc = conv_results.document

            # Debugging code that dumps the VLM predicted text in files
            if do_debug and converter_type == ConverterTypes.SMOL_DOCLING:
                debug_dir = output_dir / "debug"
                os.makedirs(debug_dir, exist_ok=True)
                if len(conv_results.pages):
                    for page_id, page in enumerate(conv_results.pages):
                        predicted_txt = page.predictions.vlm_response.text
                        if predicted_txt is None:
                            continue
                        page_text_fn = debug_dir / f"{page_hash}_{page_id}.txt"
                        with open(page_text_fn, "w") as fd:
                            fd.write(predicted_txt)

            true_doc = DoclingDocument(name=page_hash)
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_stream, true_doc=true_doc, image_scale=1.0
            )
            img = true_page_images[0]
            old_w, old_h = doc["image"].size
            old_size = Size(width=old_w, height=old_h)

            current_list = None
            labels = list(map(lambda cid: category_map[int(cid)], doc["category_id"]))
            bboxes = doc["bboxes"]
            segments = doc["pdf_cells"]
            contents = [
                " ".join(map(lambda cell: cell["text"], cells)) for cells in segments
            ]
            for l, b, c in zip(labels, bboxes, contents):
                update(true_doc, current_list, img, old_size, l, b, c)

            if do_viz:
                save_comparison_html_with_clusters(
                    filename=viz_dir / f"{true_doc.name}-clusters.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=img,
                    true_labels=TRUE_HTML_EXPORT_LABELS,
                    pred_labels=PRED_HTML_EXPORT_LABELS,
                    draw_reading_order=False,  # Disable reading-order visualization
                )
            true_doc, true_pictures, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,  # pictures_column,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,  # page_images_column,
            )

            pred_doc, pred_pictures, pred_page_images = extract_images(
                document=pred_doc,
                pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,  # pictures_column,
                page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,  # page_images_column,
            )

            record = {
                BenchMarkColumns.CONVERTER_TYPE: converter_type,
                BenchMarkColumns.CONVERTER_VERSION: docling_version(),
                BenchMarkColumns.STATUS: str(conv_results.status),
                BenchMarkColumns.DOC_ID: page_hash,
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.ORIGINAL: pdf_stream.getvalue(),
                BenchMarkColumns.MIMETYPE: "image/png",
                BenchMarkColumns.MODALITIES: [
                    EvaluationModality.LAYOUT,
                    EvaluationModality.MARKDOWN_TEXT,
                ],
            }
            pdf_stream.close()
            records.append(record)
            exported_rows += 1
            if exported_rows % SHARD_SIZE == 0:
                shard_id = exported_rows // SHARD_SIZE - 1
                save_shard_to_disk(
                    items=records, dataset_path=test_dir, shard_id=shard_id
                )
                saved_shards += 1
                records = []
        except Exception as ex:
            log.error(str(ex))
            skipped_rows += 1

    if len(records) > 0:
        shard_id = exported_rows // SHARD_SIZE
        save_shard_to_disk(items=records, dataset_path=test_dir, shard_id=shard_id)
        saved_shards += 1

    if selected_ds_len > 0:
        write_datasets_info(
            name="DocLayNetV1: end-to-end",
            output_dir=output_dir,
            num_train_rows=0,
            num_test_rows=len(records) + exported_rows,
        )

    log.info(
        "Dataset len: %s. Selected range: [%s, %s]. Exported rows: %s. Skipped_rows: %s. Saved shards: %s",
        total_ds_len,
        begin_index,
        end_index,
        exported_rows,
        skipped_rows,
        saved_shards,
    )
