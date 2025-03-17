import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    PageItem,
    ProvenanceItem,
    Size,
)
from PIL import Image
from tqdm import tqdm

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.converters.hyperscalers import (
    convert_azure_output_to_docling,
    convert_google_output_to_docling,
    convert_textract_output_to_docling,
)
from docling_eval.utils.hyperscalers import hyperscaler_clients, utils


def process_image_with_services(
    image_path: Path,
    output_base_dir: Path,
    services_to_run: List[str],
    clients: Dict,
) -> Dict:
    """Processes a single image with specified OCR services and saves outputs."""
    sample_id = image_path.parent.name
    results = {}

    service_dirs = {
        "aws": output_base_dir / sample_id / "textract_results",
        "google": output_base_dir / sample_id / "google_results",
        "azure": output_base_dir / sample_id / "azure_results",
    }
    converted_dir = output_base_dir / sample_id / "converted_results"

    dirs_to_create = list(service_dirs.values()) + [converted_dir]
    utils.create_directories(dirs_to_create)

    image_content = utils.read_image_content(image_path)

    if "aws" in services_to_run:
        results["aws"] = process_with_service(
            "aws",
            sample_id,
            image_content,
            image_path,
            service_dirs["aws"],
            converted_dir,
            lambda: hyperscaler_clients.process_with_textract(
                clients["textract"], image_content
            ),
            convert_textract_output_to_docling,
        )

    if "google" in services_to_run:
        results["google"] = process_with_service(
            "google",
            sample_id,
            image_content,
            image_path,
            service_dirs["google"],
            converted_dir,
            lambda: hyperscaler_clients.process_with_google(
                clients["google"],
                clients["google_processor_name"],
                image_content,
                "image/tiff",
            ),
            convert_google_output_to_docling,
        )

    if "azure" in services_to_run:
        results["azure"] = process_with_service(
            "azure",
            sample_id,
            image_content,
            image_path,
            service_dirs["azure"],
            converted_dir,
            lambda: hyperscaler_clients.process_with_azure(
                clients["azure"], image_content
            ),
            convert_azure_output_to_docling,
        )

    return results


def process_with_service(
    service_name: str,
    sample_id: str,
    image_content: bytes,
    image_path: Path,
    service_dir: Path,
    converted_dir: Path,
    api_call_func,
    convert_func,
) -> str:
    """Helper function to process an image with a specific OCR service."""
    output_raw_path = service_dir / f"{sample_id}_{service_name}_raw.json"
    output_converted_path = converted_dir / f"{sample_id}_{service_name}_converted.json"

    service_display_names = {
        "aws": "Textract",
        "google": "Google Doc AI",
        "azure": "Azure Document Intelligence",
    }
    display_name = service_display_names.get(service_name, service_name)

    try:
        response = api_call_func()
        if response:
            utils.write_json_output(output_raw_path, response)
            converted_output = convert_func(response, image_path)
            utils.write_json_output(output_converted_path, converted_output)
            print(f"✓ {display_name} processing complete for {sample_id}")
            return "success"
        else:
            print(
                f"{display_name} processing failed for {sample_id} (API might have returned None)"
            )
            return "error"
    except Exception as e:
        print(f"Error processing {sample_id} with {display_name}: {str(e)}")
        raise e


def initialize_clients(reprocess_hyperscalers: bool) -> Dict:
    """Initialize hyperscaler clients only if reprocessing is requested."""
    if not reprocess_hyperscalers:
        return {}

    clients = {}
    clients["textract"] = hyperscaler_clients.initialize_textract_client()

    clients["google"], clients["google_processor_name"] = (
        hyperscaler_clients.initialize_google_doc_ai_client()
    )

    clients["azure"] = (
        hyperscaler_clients.initialize_azure_document_intelligence_client()
    )

    return clients


def load_converted_results(doc_id: str, input_dir: Path) -> Dict:
    """Load converted results from hyperscaler services if they exist."""
    hyperscaler_predictions = {}
    converted_results_dir = input_dir / doc_id / "converted_results"

    services = [
        ("aws_prediction", "aws"),
        ("google_prediction", "google"),
        ("azure_prediction", "azure"),
    ]

    for prediction_key, service_name in services:
        converted_path = (
            converted_results_dir / f"{doc_id}_{service_name}_converted.json"
        )
        print(converted_path)
        if converted_path.exists():
            with open(converted_path, "r") as f:
                hyperscaler_predictions[prediction_key] = json.load(f)

    return hyperscaler_predictions


def create_docling_document(
    doc_id: str, gt_data: Dict, image_file: Path
) -> DoclingDocument:
    """Create a DoclingDocument from ground truth data and image file."""
    w, h = Image.open(image_file).size

    true_doc = DoclingDocument(name=doc_id)
    true_doc.pages[1] = PageItem(size=Size(width=float(w), height=float(h)), page_no=1)

    for page_idx, page in enumerate(gt_data["pages"], 1):
        for text, bbox, _ in zip(page["text"], page["bbox"], page["score"]):
            bbox_obj = BoundingBox.from_tuple(
                (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0] + bbox[2]),
                    float(bbox[1] + bbox[3]),
                ),
                CoordOrigin.TOPLEFT,
            )
            prov = ProvenanceItem(
                page_no=page_idx, bbox=bbox_obj, charspan=(0, len(text))
            )
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

    return true_doc


def create(
    name: str,
    split: str,
    input_dir: Path,
    output_dir: Path,
    do_viz: bool = False,
    max_items: int = -1,
    reprocess_hyperscalers: bool = False,
) -> None:
    """Processes ground truth files, runs OCR services, and saves results."""
    # TODO: Upload the dataset to Huggingface and then use this code to create the dataset
    # ds = load_dataset(name, split=split)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization directory if needed
    if do_viz:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_files = list(input_dir.rglob("ground_truth.json"))
    if max_items > 0:
        ground_truth_files = ground_truth_files[:max_items]

    clients = initialize_clients(
        reprocess_hyperscalers
    )  # clients only if reprocessing is requested
    services_to_run = ["aws", "google", "azure"] if reprocess_hyperscalers else []

    count = 0
    records = []

    for gt_file in tqdm(ground_truth_files):
        try:
            image_file = gt_file.parent / "original.tif"
            if not image_file.exists():
                print(f"Warning: No image file found for {gt_file}")
                continue

            doc_id = gt_file.parent.name

            if reprocess_hyperscalers:
                print(f"Re-running hyperscalers for {doc_id}...")
                process_image_with_services(
                    image_file, input_dir, services_to_run, clients
                )

            with open(gt_file, "r") as f:
                gt_data = json.load(f)

            # Create DoclingDocument from ground truth
            true_doc = create_docling_document(doc_id, gt_data, image_file)

            # Create record
            record = {
                BenchMarkColumns.DOC_ID: doc_id,
                BenchMarkColumns.GROUNDTRUTH: true_doc.export_to_dict(),
                # "mimetype": "image/tiff", # TODO: Do we need to add the mimetype?
            }

            # Load converted results
            hyperscaler_predictions = load_converted_results(doc_id, input_dir)
            record.update(hyperscaler_predictions)
            records.append(record)

            # Generate visualization if requested
            if do_viz:
                viz_path = viz_dir / f"{doc_id}_viz.html"
                # TODO: Add visualization code

            # Save results periodically
            count += 1
            if count % 1000 == 0:
                shard_id = count // 1000 - 1
                utils.save_shard_to_disk(records, output_dir, shard_id)
                records = []

        except Exception as e:
            print(f"Error processing {gt_file}: {str(e)}")
            continue

    # Save any remaining records
    if records:
        shard_id = count // 1000
        utils.save_shard_to_disk(records, output_dir, shard_id)

    # Write dataset info
    utils.write_dataset_info(
        name="Ground Truth Dataset",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=count,
    )
