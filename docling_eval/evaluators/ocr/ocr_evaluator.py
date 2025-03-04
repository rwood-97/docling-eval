import glob
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate
from datasets import load_dataset
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns  # type: ignore


class DatasetStatistics(BaseModel):
    mean: Dict[str, float]
    median: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]
    std: Dict[str, float]
    count: int


class PageEvaluation(BaseModel):
    doc_id: str
    gd_text: str
    extracted_texts: Dict[str, str]
    cers: Dict[str, float]


class DatasetTextractEvaluation(BaseModel):
    evaluations: List[PageEvaluation]
    cer_stats: DatasetStatistics


def compute_stats(cers_list: List[Dict[str, float]]) -> DatasetStatistics:
    """Compute statistics for CER scores across hyperscalers"""
    import numpy as np

    providers = ["aws", "google", "azure"]
    result = {
        "mean": {},
        "median": {},
        "min": {},
        "max": {},
        "std": {},
        "count": len(cers_list),
    }

    for provider in providers:
        values = [item[provider] for item in cers_list if provider in item]
        if values:
            result["mean"][provider] = float(np.mean(values))
            result["median"][provider] = float(np.median(values))
            result["min"][provider] = float(np.min(values))
            result["max"][provider] = float(np.max(values))
            result["std"][provider] = float(np.std(values))

    return DatasetStatistics(**result)


def load_jsonl_dataset(jsonl_path: str) -> List[Dict]:
    """Load dataset from JSONL file(s)"""
    data = []
    for file_path in glob.glob(jsonl_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


class OCREvaluator:
    def __init__(self):
        # https://huggingface.co/spaces/evaluate-metric/cer
        self._cer_eval = evaluate.load("cer")

    def __call__(
        self, ds_path: Path, output_path: Path, split: str = "test"
    ) -> DatasetTextractEvaluation:
        jsonl_files = str(ds_path / "*.jsonl")
        logging.info(f"Loading JSONL files from pattern: {jsonl_files}")

        dataset = load_jsonl_dataset(jsonl_files)
        if not dataset:
            raise ValueError(f"Failed to load dataset from {jsonl_files}")

        logging.info(f"Loaded {len(dataset)} records from JSONL files")

        evaluations: List[PageEvaluation] = []
        cers_list = []

        for i, data in tqdm(
            enumerate(dataset),
            desc="Hyperscaler OCR evaluations",
            ncols=120,
            total=len(dataset),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            # TODO: Fix bug in true_doc_dict conversion
            # true_doc: DoclingDocument = DoclingDocument.model_validate_json(
            #     true_doc_dict
            # )
            true_doc = true_doc_dict

            hyperscalers = ["aws", "google", "azure"]
            pred_docs = {}
            extracted_texts = {}

            for provider in hyperscalers:
                pred_key = f"{provider}_prediction"
                if pred_key in data:
                    pred_doc_dict = data[pred_key]
                    # TODO: fix bug in pred_doc_dict conversion
                    # pred_docs[provider] = DoclingDocument.model_validate_json(
                    #     pred_doc_dict
                    # )
                    pred_docs[provider] = pred_doc_dict

            # Extract text from documents
            # ToDO: Fix bug in true_doc conversion
            # gd_text = self._extract_text(true_doc.export_to_dict())
            gd_text = self._extract_text(true_doc)
            cers = {}

            for provider, pred_doc in pred_docs.items():
                # extracted_text = self._extract_text(pred_doc.export_to_dict())
                extracted_text = self._extract_text(pred_doc)
                extracted_texts[provider] = extracted_text

                if gd_text and extracted_text:
                    cers[provider] = self._compute_cer_score(gd_text, extracted_text)
                else:
                    cers[provider] = 1.0  # max error when text is missing

            cers_list.append(cers)
            text_evaluation = PageEvaluation(
                doc_id=doc_id,
                gd_text=gd_text,
                extracted_texts=extracted_texts,
                cers=cers,
            )
            evaluations.append(text_evaluation)

            # Print CER for each document and hyperscaler
            print(f"Document {doc_id} CER scores:")
            for provider, cer_value in cers.items():
                print(f"  - {provider.upper()}: {cer_value:.4f}")

        # Compute statistics across all documents
        cer_stats = compute_stats(cers_list)

        # Print overall CER statistics
        print("\nOverall CER Statistics:")
        for provider in ["aws", "google", "azure"]:
            if provider in cer_stats.mean:
                print(f"\n{provider.upper()} Statistics:")
                print(f"  - Mean CER: {cer_stats.mean[provider]:.4f}")
                print(f"  - Median CER: {cer_stats.median[provider]:.4f}")
                print(f"  - Min CER: {cer_stats.min[provider]:.4f}")
                print(f"  - Max CER: {cer_stats.max[provider]:.4f}")
                print(f"  - Std Dev: {cer_stats.std[provider]:.4f}")

        output_path.mkdir(parents=True, exist_ok=True)
        results_path = output_path / "evaluation_results.jsonl"
        with open(results_path, "w") as f:
            stats_json = json.dumps({"type": "statistics", "data": cer_stats.dict()})
            f.write(stats_json + "\n")
            for eval_item in evaluations:
                eval_json = json.dumps({"type": "evaluation", "data": eval_item.dict()})
                f.write(eval_json + "\n")

        logging.info(f"Saved evaluation results to {results_path}")

        ds_text_evaluations = DatasetTextractEvaluation(
            evaluations=evaluations, cer_stats=cer_stats
        )
        return ds_text_evaluations

    def _compute_cer_score(self, true_txt: str, pred_txt: str):
        """
        Compute CER score with the HF evaluate and the default Tokenizer
        """
        result = self._cer_eval.compute(predictions=[pred_txt], references=[true_txt])
        print(f"CER: {result}")
        # cer = result["cer"]
        # return cer
        return result

    def _extract_text(self, json_data: Dict[str, Any]) -> str:
        """
        Extract text from document JSON structure
        """
        extracted_text = ""
        if "texts" in json_data:
            for text_item in json_data["texts"]:
                if "text" in text_item:
                    extracted_text += text_item["text"] + " "
        return extracted_text.strip()

