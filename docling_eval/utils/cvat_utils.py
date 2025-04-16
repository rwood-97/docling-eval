import logging
from typing import Dict, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument, TableData, TableItem

# Configure logging
_log = logging.getLogger(__name__)


def find_table_data(doc: DoclingDocument, prov, iou_cutoff: float = 0.90):
    """
    Find table data in a document based on provenance.

    Args:
        doc: Document to search in
        prov: Provenance to match
        iou_cutoff: IoU threshold for matching

    Returns:
        TableData structure from the matching table or an empty structure
    """
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            for item_prov in item.prov:
                if item_prov.page_no != prov.page_no:
                    continue

                page_height = doc.pages[item_prov.page_no].size.height

                iou = item_prov.bbox.intersection_over_union(prov.bbox)

                if iou > iou_cutoff:
                    _log.info(f"Found matching table data with IoU: {iou:.2f}")
                    return item.data

    _log.warning("No matching table data found")

    # Return empty table data
    return TableData(num_rows=-1, num_cols=-1, table_cells=[])
