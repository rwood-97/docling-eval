import json
import sys
import uuid
from pathlib import Path


def convert_textract_output(response, image_path: Path):
    """Converts Textract output to the desired format."""
    converted_format = {
        "schema_name": "DoclingDocument",
        "version": "1.0",
        "name": Path(image_path).stem,
        "origin": {
            "mimetype": "image/tiff",
            "binary_hash": hash(json.dumps(response)),
            "filename": image_path.name,
        },
        "furniture": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "furniture",
            "label": "furniture",
        },
        "body": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "body",
            "label": "body",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "pages": {},
    }
    page_dims = {}
    page_no = str(response.get("DocumentMetadata", {}).get("Pages", 1))

    for block in response.get("Blocks", []):
        if block["BlockType"] == "PAGE":
            # page_no = str(block["Page"]) # NOTE: somehow the page number is missing in the new response where as it was available in the old response
            page_dims[page_no] = {
                "width": block["Geometry"]["BoundingBox"]["Width"],
                "height": block["Geometry"]["BoundingBox"]["Height"],
            }
            converted_format["pages"][page_no] = {
                "size": {
                    "width": block["Geometry"]["BoundingBox"]["Width"],
                    "height": block["Geometry"]["BoundingBox"]["Height"],
                },
                "page_no": page_no,
            }
    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            text_block = {
                "self_ref": str(uuid.uuid4()),
                "parent": {"$ref": converted_format["body"]["self_ref"]},
                "children": [],
                "label": "text_line",
                "prov": [
                    {
                        # "page_no": block["Page"],
                        "page_no": page_no,
                        "bbox": {
                            "l": block["Geometry"]["BoundingBox"]["Left"],
                            "t": block["Geometry"]["BoundingBox"]["Top"],
                            "r": block["Geometry"]["BoundingBox"]["Left"]
                            + block["Geometry"]["BoundingBox"]["Width"],
                            "b": block["Geometry"]["BoundingBox"]["Top"]
                            + block["Geometry"]["BoundingBox"]["Height"],
                            "coord_origin": "top-left",
                        },
                        "charspan": (
                            [block["Text"][0], len(block["Text"])]
                            if "Text" in block
                            else []
                        ),  # dummy charspan
                    }
                ],
                "orig": block.get("Text", ""),
                "text": block.get("Text", ""),
                "level": 1,
            }
            converted_format["texts"].append(text_block)
        elif block["BlockType"] == "TABLE":
            table_block = {
                "self_ref": str(uuid.uuid4()),
                "parent": {"$ref": converted_format["body"]["self_ref"]},
                "children": [],
                "label": "table",
                "prov": [
                    {
                        # TODO: Need to fix this.
                        "page_no": page_no,
                        "bbox": {
                            "l": block["Geometry"]["BoundingBox"]["Left"],
                            "t": block["Geometry"]["BoundingBox"]["Top"],
                            "r": block["Geometry"]["BoundingBox"]["Left"]
                            + block["Geometry"]["BoundingBox"]["Width"],
                            "b": block["Geometry"]["BoundingBox"]["Top"]
                            + block["Geometry"]["BoundingBox"]["Height"],
                            "coord_origin": "top-left",
                        },
                        "charspan": [],  # dummy charspan
                    }
                ],
                "captions": [],
                "references": [],
                "footnotes": [],
                "data": {
                    "table_cells": [],
                    "num_rows": 0,  # needs calculation
                    "num_cols": 0,  # needs calculation
                    "grid": [],  # complex, skipping for now
                },
            }
            cell_blocks = [
                b
                for b in response.get("Blocks", [])
                if b["BlockType"] == "CELL" and b["ParentId"] == block["Id"]
            ]
            for cell_block in cell_blocks:
                cell_text = ""
                line_blocks_in_cell = [
                    b
                    for b in response.get("Blocks", [])
                    if b["BlockType"] == "LINE" and b["ParentId"] == cell_block["Id"]
                ]
                for line_block in line_blocks_in_cell:
                    cell_text += line_block.get("Text", "") + " "

                table_cell = {
                    "bbox": {
                        "l": cell_block["Geometry"]["BoundingBox"]["Left"],
                        "t": cell_block["Geometry"]["BoundingBox"]["Top"],
                        "r": cell_block["Geometry"]["BoundingBox"]["Left"]
                        + cell_block["Geometry"]["BoundingBox"]["Width"],
                        "b": cell_block["Geometry"]["BoundingBox"]["Top"]
                        + cell_block["Geometry"]["BoundingBox"]["Height"],
                        "coord_origin": "top-left",
                    },
                    "row_span": 1,  # Needs better logic
                    "col_span": 1,  # Needs better logic
                    "start_row_offset_idx": 0,  # Needs better logic
                    "end_row_offset_idx": 0,  # Needs better logic
                    "start_col_offset_idx": 0,  # Needs better logic
                    "end_col_offset_idx": 0,  # Needs better logic
                    "text": cell_text.strip(),
                    "column_header": False,  # Needs better logic
                    "row_header": False,  # Needs better logic
                    "row_section": False,  # Needs better logic
                }
                table_block["data"]["table_cells"].append(table_cell)
            converted_format["tables"].append(table_block)
    return converted_format


def convert_google_output(document, image_path):
    """Converts Google Document AI output to the desired format."""
    import json
    import uuid
    from pathlib import Path

    converted_format = {
        "schema_name": "DoclingDocument",
        "version": "1.0",
        "name": Path(image_path).stem,
        "origin": {
            "mimetype": "image/tiff",
            "binary_hash": hash(json.dumps(document)),
            "filename": Path(image_path).name,
        },
        "furniture": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "furniture",
            "label": "furniture",
        },
        "body": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "body",
            "label": "body",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "pages": {},
    }

    for page in document.get("pages", []):
        page_no = str(page.get("pageNumber", "1"))
        converted_format["pages"][page_no] = {
            "size": {
                "width": page.get("dimension", {}).get("width", 0),
                "height": page.get("dimension", {}).get("height", 0),
            },
            "page_no": page.get("pageNumber", 1),
        }

        # Process blocks (they are at the page level in Google DocAI)
        for block in page.get("blocks", []):
            # Extract bounding poly for the block
            block_bbox = {"l": 0, "t": 0, "r": 0, "b": 0, "coord_origin": "top-left"}

            if (
                "layout" in block
                and "boundingPoly" in block["layout"]
                and "vertices" in block["layout"]["boundingPoly"]
            ):
                vertices = block["layout"]["boundingPoly"]["vertices"]
                if len(vertices) >= 4:
                    block_bbox = {
                        "l": vertices[0].get("x", 0),
                        "t": vertices[0].get("y", 0),
                        "r": vertices[2].get("x", 0),
                        "b": vertices[2].get("y", 0),
                        "coord_origin": "top-left",
                    }

        # Process paragraphs (they are at the page level in Google DocAI)
        for paragraph in page.get("paragraphs", []):
            # Get text content from the text_anchor and text_segments
            text_content = ""
            if "layout" in paragraph and "textAnchor" in paragraph["layout"]:
                for text_segment in paragraph["layout"]["textAnchor"].get(
                    "textSegments", []
                ):
                    # Get the text using the segment's indices
                    if "endIndex" in text_segment:
                        start_index = int(text_segment.get("startIndex", 0))
                        end_index = int(text_segment.get("endIndex", 0))
                        if document.get("text") and start_index < len(document["text"]):
                            text_content += document["text"][start_index:end_index]

            # Get bounding box for paragraph
            para_bbox = {"l": 0, "t": 0, "r": 0, "b": 0, "coord_origin": "top-left"}

            if (
                "layout" in paragraph
                and "boundingPoly" in paragraph["layout"]
                and "vertices" in paragraph["layout"]["boundingPoly"]
            ):
                vertices = paragraph["layout"]["boundingPoly"]["vertices"]
                if len(vertices) >= 4:
                    para_bbox = {
                        "l": vertices[0].get("x", 0),
                        "t": vertices[0].get("y", 0),
                        "r": vertices[2].get("x", 0),
                        "b": vertices[2].get("y", 0),
                        "coord_origin": "top-left",
                    }

            text_block = {
                "self_ref": str(uuid.uuid4()),
                "parent": {"$ref": converted_format["body"]["self_ref"]},
                "children": [],
                "label": "text_paragraph",
                "prov": [
                    {
                        "page_no": page.get("pageNumber", 1),
                        "bbox": para_bbox,
                        "charspan": [],  # Dummy charspan
                    }
                ],
                "orig": text_content,
                "text": text_content,
                "level": 2,  # paragraph level
            }
            converted_format["texts"].append(text_block)

        # Process tables
        for table in page.get("tables", []):
            # Get table bounding box
            table_bbox = {"l": 0, "t": 0, "r": 0, "b": 0, "coord_origin": "top-left"}

            if (
                "layout" in table
                and "boundingPoly" in table["layout"]
                and "vertices" in table["layout"]["boundingPoly"]
            ):
                vertices = table["layout"]["boundingPoly"]["vertices"]
                if len(vertices) >= 4:
                    table_bbox = {
                        "l": vertices[0].get("x", 0),
                        "t": vertices[0].get("y", 0),
                        "r": vertices[2].get("x", 0),
                        "b": vertices[2].get("y", 0),
                        "coord_origin": "top-left",
                    }

            table_block = {
                "self_ref": str(uuid.uuid4()),
                "parent": {"$ref": converted_format["body"]["self_ref"]},
                "children": [],
                "label": "table",
                "prov": [
                    {
                        "page_no": page.get("pageNumber", 1),
                        "bbox": table_bbox,
                        "charspan": [],  # dummy charspan
                    }
                ],
                "captions": [],
                "references": [],
                "footnotes": [],
                "data": {
                    "table_cells": [],
                    "num_rows": len(table.get("headerRows", []))
                    + len(table.get("bodyRows", [])),
                    "num_cols": 0,  # Will calculate based on cells
                    "grid": [],  # complex, skipping for now
                },
            }

            # Process header rows
            for row_index, row in enumerate(table.get("headerRows", [])):
                max_columns = max(
                    table_block["data"]["num_cols"], len(row.get("cells", []))
                )
                table_block["data"]["num_cols"] = max_columns

                for cell_index, cell in enumerate(row.get("cells", [])):
                    cell_text_content = ""
                    if "layout" in cell and "textAnchor" in cell["layout"]:
                        for text_segment in cell["layout"]["textAnchor"].get(
                            "textSegments", []
                        ):
                            start_index = int(text_segment.get("startIndex", 0))
                            end_index = int(text_segment.get("endIndex", 0))
                            if document.get("text") and start_index < len(
                                document["text"]
                            ):
                                cell_text_content += document["text"][
                                    start_index:end_index
                                ]

                    # Get cell bounding box
                    cell_bbox = {
                        "l": 0,
                        "t": 0,
                        "r": 0,
                        "b": 0,
                        "coord_origin": "top-left",
                    }

                    if (
                        "layout" in cell
                        and "boundingPoly" in cell["layout"]
                        and "vertices" in cell["layout"]["boundingPoly"]
                    ):
                        vertices = cell["layout"]["boundingPoly"]["vertices"]
                        if len(vertices) >= 4:
                            cell_bbox = {
                                "l": vertices[0].get("x", 0),
                                "t": vertices[0].get("y", 0),
                                "r": vertices[2].get("x", 0),
                                "b": vertices[2].get("y", 0),
                                "coord_origin": "top-left",
                            }

                    table_cell = {
                        "bbox": cell_bbox,
                        "row_span": cell.get("rowSpan", 1),
                        "col_span": cell.get("colSpan", 1),
                        "start_row_offset_idx": row_index,
                        "end_row_offset_idx": row_index + cell.get("rowSpan", 1) - 1,
                        "start_col_offset_idx": cell_index,
                        "end_col_offset_idx": cell_index + cell.get("colSpan", 1) - 1,
                        "text": cell_text_content.strip(),
                        "column_header": True,  # This is a header row
                        "row_header": False,
                        "row_section": False,
                    }
                    table_block["data"]["table_cells"].append(table_cell)

            # Process body rows
            header_row_count = len(table.get("headerRows", []))
            for row_index, row in enumerate(table.get("bodyRows", [])):
                actual_row_index = header_row_count + row_index
                max_columns = max(
                    table_block["data"]["num_cols"], len(row.get("cells", []))
                )
                table_block["data"]["num_cols"] = max_columns

                for cell_index, cell in enumerate(row.get("cells", [])):
                    cell_text_content = ""
                    if "layout" in cell and "textAnchor" in cell["layout"]:
                        for text_segment in cell["layout"]["textAnchor"].get(
                            "textSegments", []
                        ):
                            start_index = int(text_segment.get("startIndex", 0))
                            end_index = int(text_segment.get("endIndex", 0))
                            if document.get("text") and start_index < len(
                                document["text"]
                            ):
                                cell_text_content += document["text"][
                                    start_index:end_index
                                ]

                    # Get cell bounding box
                    cell_bbox = {
                        "l": 0,
                        "t": 0,
                        "r": 0,
                        "b": 0,
                        "coord_origin": "top-left",
                    }

                    if (
                        "layout" in cell
                        and "boundingPoly" in cell["layout"]
                        and "vertices" in cell["layout"]["boundingPoly"]
                    ):
                        vertices = cell["layout"]["boundingPoly"]["vertices"]
                        if len(vertices) >= 4:
                            cell_bbox = {
                                "l": vertices[0].get("x", 0),
                                "t": vertices[0].get("y", 0),
                                "r": vertices[2].get("x", 0),
                                "b": vertices[2].get("y", 0),
                                "coord_origin": "top-left",
                            }

                    table_cell = {
                        "bbox": cell_bbox,
                        "row_span": cell.get("rowSpan", 1),
                        "col_span": cell.get("colSpan", 1),
                        "start_row_offset_idx": actual_row_index,
                        "end_row_offset_idx": actual_row_index
                        + cell.get("rowSpan", 1)
                        - 1,
                        "start_col_offset_idx": cell_index,
                        "end_col_offset_idx": cell_index + cell.get("colSpan", 1) - 1,
                        "text": cell_text_content.strip(),
                        "column_header": False,
                        "row_header": cell_index
                        == 0,  # First column might be row header
                        "row_section": False,
                    }
                    table_block["data"]["table_cells"].append(table_cell)

            converted_format["tables"].append(table_block)

    return converted_format


def convert_azure_output(analyze_result, image_path):
    """Converts Azure Form Recognizer output to the desired format."""
    import json
    import uuid
    from pathlib import Path

    converted_format = {
        "schema_name": "DoclingDocument",
        "version": "1.0",
        "name": Path(image_path).stem,
        "origin": {
            "mimetype": "image/tiff",
            "binary_hash": hash(json.dumps(analyze_result)),
            "filename": Path(image_path).name,
        },
        "furniture": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "furniture",
            "label": "furniture",
        },
        "body": {
            "self_ref": str(uuid.uuid4()),
            "children": [],
            "name": "body",
            "label": "body",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "pages": {},
    }
    page_map = {}
    for page in analyze_result.get("pages", []):
        page_no = str(page.get("page_number", "1"))
        page_map[page.get("page_number")] = str(uuid.uuid4())
        converted_format["pages"][page_no] = {
            "size": {
                "width": page.get("width", 0),
                "height": page.get("height", 0),
            },
            "page_no": page.get("page_number", 1),
        }
        for word in page.get("words", []):
            # Handle polygon as a flat array of coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
            # or as coordinates already parsed into a list of points
            polygon = word.get("polygon", [])

            # Extract coordinates based on the format of polygon
            if polygon and isinstance(polygon, list):
                if len(polygon) >= 8 and all(
                    isinstance(p, (int, float)) for p in polygon
                ):
                    # Flat array format: [x1, y1, x2, y2, x3, y3, x4, y4]
                    l, t = polygon[0], polygon[1]
                    r, b = (
                        polygon[4],
                        polygon[5],
                    )  # Using bottom-right corner (3rd point)
                elif len(polygon) >= 4 and all(
                    isinstance(p, dict) and "x" in p and "y" in p for p in polygon
                ):
                    # Array of point objects: [{x, y}, {x, y}, ...]
                    l, t = polygon[0]["x"], polygon[0]["y"]
                    r, b = polygon[2]["x"], polygon[2]["y"]
                else:
                    # if unknown format, use defaults
                    l, t, r, b = 0, 0, 0, 0
            else:
                l, t, r, b = 0, 0, 0, 0

            text_block = {
                "self_ref": str(uuid.uuid4()),
                "parent": {
                    "$ref": (
                        page_map[page.get("page_number")]
                        if page.get("page_number") in page_map
                        else converted_format["body"]["self_ref"]
                    )
                },  # associating text with page
                "children": [],
                "label": "word",
                "prov": [
                    {
                        "page_no": page.get("page_number", 1),
                        "bbox": {
                            "l": l,
                            "t": t,
                            "r": r,
                            "b": b,
                            "coord_origin": "top-left",
                        },
                        "charspan": [],  # Dummy charspan
                    }
                ],
                "orig": word.get("content", ""),
                "text": word.get("content", ""),
                "level": 3,  # word level
            }
            converted_format["texts"].append(text_block)

    for table_idx, table in enumerate(analyze_result.get("tables", [])):
        # similar fix for table bounding regions
        table_bounds = (
            table.get("boundingRegions", [{}])[0]
            if table.get("boundingRegions")
            else {}
        )
        table_polygon = table_bounds.get("polygon", [])

        # Extract coordinates based on the format of polygon
        if table_polygon and isinstance(table_polygon, list):
            if len(table_polygon) >= 8 and all(
                isinstance(p, (int, float)) for p in table_polygon
            ):
                # Flat array format
                t_l, t_t = table_polygon[0], table_polygon[1]
                t_r, t_b = table_polygon[4], table_polygon[5]
            elif len(table_polygon) >= 4 and all(
                isinstance(p, dict) and "x" in p and "y" in p for p in table_polygon
            ):
                # Array of point objects
                t_l, t_t = table_polygon[0]["x"], table_polygon[0]["y"]
                t_r, t_b = table_polygon[2]["x"], table_polygon[2]["y"]
            else:
                t_l, t_t, t_r, t_b = 0, 0, 0, 0
        else:
            t_l, t_t, t_r, t_b = 0, 0, 0, 0

        table_block = {
            "self_ref": str(uuid.uuid4()),
            "parent": {
                "$ref": (
                    page_map[table.get("page_range", {}).get("first_page_number")]
                    if table.get("page_range", {}).get("first_page_number") in page_map
                    else converted_format["body"]["self_ref"]
                )
            },  # associating table with page
            "children": [],
            "label": "table",
            "prov": [
                {
                    "page_no": table.get("page_range", {}).get("first_page_number", 1),
                    "bbox": {
                        "l": t_l,
                        "t": t_t,
                        "r": t_r,
                        "b": t_b,
                        "coord_origin": "top-left",
                    },
                    "charspan": [],  # dummy charspan
                }
            ],
            "captions": [],
            "references": [],
            "footnotes": [],
            "data": {
                "table_cells": [],
                "num_rows": table.get("row_count", 0),
                "num_cols": table.get("column_count", 0),
                "grid": [],  # complex, skipping for now
            },
        }

        for cell in table.get("cells", []):
            cell_text_content = cell.get("content", "")

            # Fix for cell bounding regions
            cell_bounds = (
                cell.get("boundingRegions", [{}])[0]
                if cell.get("boundingRegions")
                else {}
            )
            cell_polygon = cell_bounds.get("polygon", [])

            # Extract coordinates based on the format of polygon
            if cell_polygon and isinstance(cell_polygon, list):
                if len(cell_polygon) >= 8 and all(
                    isinstance(p, (int, float)) for p in cell_polygon
                ):
                    # Flat array format
                    c_l, c_t = cell_polygon[0], cell_polygon[1]
                    c_r, c_b = cell_polygon[4], cell_polygon[5]
                elif len(cell_polygon) >= 4 and all(
                    isinstance(p, dict) and "x" in p and "y" in p for p in cell_polygon
                ):
                    # Array of point objects
                    c_l, c_t = cell_polygon[0]["x"], cell_polygon[0]["y"]
                    c_r, c_b = cell_polygon[2]["x"], cell_polygon[2]["y"]
                else:
                    c_l, c_t, c_r, c_b = 0, 0, 0, 0
            else:
                c_l, c_t, c_r, c_b = 0, 0, 0, 0

            table_cell = {
                "bbox": {
                    "l": c_l,
                    "t": c_t,
                    "r": c_r,
                    "b": c_b,
                    "coord_origin": "top-left",
                },
                "row_span": cell.get("row_span", 1),
                "col_span": cell.get("column_span", 1),
                "start_row_offset_idx": cell.get("row_index", 0),
                "end_row_offset_idx": cell.get("row_index", 0)
                + cell.get("row_span", 1)
                - 1,
                "start_col_offset_idx": cell.get("column_index", 0),
                "end_col_offset_idx": cell.get("column_index", 0)
                + cell.get("column_span", 1)
                - 1,
                "text": cell_text_content.strip(),
                "column_header": False,  # Needs better logic
                "row_header": False,  # Needs better logic
                "row_section": False,  # Needs better logic
            }
            table_block["data"]["table_cells"].append(table_cell)
        converted_format["tables"].append(table_block)
    return converted_format
