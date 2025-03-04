# utils.py

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from PIL import Image


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle bytes and Pydantic models"""

    def default(self, obj):
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        # Check for both v1 and v2 Pydantic models.
        if hasattr(obj, "model_dump"):  # For Pydantic v2 models
            return obj.model_dump()
        if hasattr(obj, "dict"):  # For Pydantic v1 models
            return obj.dict()
        return super().default(obj)


def create_directories(dir_paths: List[Path]):
    """Creates directories if they don't exist."""
    for dir_path in dir_paths:
        dir_path.mkdir(parents=True, exist_ok=True)


def read_image_content(image_path: Path) -> bytes:
    """Reads image content in binary mode."""
    with open(image_path, "rb") as image:
        return image.read()


def write_json_output(output_path: Path, data: Dict):
    """Writes JSON data to a file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_shard_to_disk(items: List[Dict], dataset_path: Path, shard_id: int) -> None:
    """Saves a shard of records to disk using custom JSON encoder."""
    shard_path = dataset_path / f"shard_{shard_id:05d}.jsonl"
    with open(shard_path, "w") as f:
        for item in items:
            json_str = json.dumps(item, cls=CustomJSONEncoder)
            f.write(json_str + "\n")


def write_dataset_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
) -> None:
    """Writes dataset information to a file."""
    info = {
        "name": name,
        "num_train_rows": num_train_rows,
        "num_test_rows": num_test_rows,
        "creation_date": str(datetime.now()),
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)


def convert_image_to_bytes(image_path: Path) -> bytes:
    """Loads an image and converts it to bytes."""
    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="TIFF")  # Or appropriate format
        return img_byte_arr.getvalue()
