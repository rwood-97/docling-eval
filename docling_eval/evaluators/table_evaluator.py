import glob
import logging
import os
import statistics
import time
from pathlib import Path
from typing import List, Optional, Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import DoclingDocument, TableItem
from lxml import html
from pydantic import BaseModel, ValidationError, model_validator
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.utils.teds import TEDScorer

_log = logging.getLogger(__name__)


class TableEvaluation(BaseModel):
    filename: str = "<unknown>"
    table_id: int = -1
    TEDS: float
    is_complex: bool = False

    true_ncols: int = -1
    pred_ncols: int = -1

    true_nrows: int = -1
    pred_nrows: int = -1


class DatasetStatistics(BaseModel):
    total: int

    mean: float
    median: float
    std: float

    bins: List[float]
    hist: List[float]

    @model_validator(mode="after")
    def check_bins_and_hist_lengths(cls, values):
        if len(values.bins) != len(values.hist) + 1:
            raise ValueError("`bins` must have exactly one more element than `hist`.")
        return values

    def to_table(self) -> Tuple[List[List[str]], List[str]]:

        headers = ["x0<=TEDS", "TEDS<=x1", "%", "total"]

        # Calculate bin widths
        bin_widths = [
            self.bins[i + 1] - self.bins[i] for i in range(len(self.bins) - 1)
        ]
        bin_middle = [
            (self.bins[i + 1] + self.bins[i]) / 2.0 for i in range(len(self.bins) - 1)
        ]

        table = []
        for i in range(len(self.bins) - 1):
            table.append(
                [
                    f"{self.bins[i+0]:.3f}",
                    f"{self.bins[i+1]:.3f}",
                    f"{100.0*float(self.hist[i])/float(self.total):.2f}",
                    f"{self.hist[i]}",
                ]
            )

        return table, headers

    def save_histogram(self, figname: Path):
        # Calculate bin widths
        bin_widths = [
            self.bins[i + 1] - self.bins[i] for i in range(len(self.bins) - 1)
        ]
        bin_middle = [
            (self.bins[i + 1] + self.bins[i]) / 2.0 for i in range(len(self.bins) - 1)
        ]

        """
        for i in range(len(self.bins) - 1):
            logging.info(
                f"{i:02} [{self.bins[i]:.3f}, {self.bins[i+1]:.3f}]: {self.hist[i]}"
            )
        """

        # Plot histogram
        plt.bar(bin_middle, self.hist, width=bin_widths, edgecolor="black")

        plt.xlabel("TEDS")
        plt.ylabel("Frequency")
        plt.title(
            f"mean: {self.mean:.2f}, median: {self.median:.2f}, std: {self.std:.2f}, total: {self.total}"
        )

        logging.info(f"saving figure to {figname}")
        plt.savefig(figname)


class DatasetTableEvaluation(BaseModel):
    evaluations: list[TableEvaluation]

    TEDS: DatasetStatistics
    TEDS_simple: DatasetStatistics
    TEDS_complex: DatasetStatistics


def compute_stats(values: List[float]) -> DatasetStatistics:
    total: int = len(values)

    mean: float = statistics.mean(values) if len(values) > 0 else -1
    median: float = statistics.median(values) if len(values) > 0 else -1
    std: float = statistics.stdev(values) if len(values) > 0 else -1
    logging.info(f"total: {total}, mean: {mean}, median: {median}, std: {std}")

    # Compute the histogram with 20 bins between 0 and 1
    hist, bins = np.histogram(values, bins=20, range=(0, 1))
    logging.info(f"#-hist: {len(hist)}, #-bins: {len(bins)}")

    return DatasetStatistics(
        total=total, mean=mean, median=median, std=std, hist=hist, bins=bins
    )


def is_complex_table(table: TableItem) -> bool:
    r"""
    Implement the logic to check if table is complex
    """
    for cell in table.data.table_cells:
        if cell.row_span > 1 or cell.col_span > 1:
            return True
    return False


class TableEvaluator:
    r"""
    Evaluate table predictions from HF dataset with the columns:
    """

    def __init__(self) -> None:
        self._teds_scorer = TEDScorer()
        self._stopwords = ["<i>", "</i>", "<b>", "</b>", "<u>", "</u>"]

    def __call__(self, ds_path: Path, split: str = "test") -> DatasetTableEvaluation:
        r"""
        Load a dataset in HF format. Expected columns with DoclingDocuments
        "GTDoclingDocument"
        "PredictionDoclingDocument"
        """
        logging.info(f"loading from: {ds_path}")

        # Load the Parquet file
        # dataset = Dataset.from_parquet("benchmarks/dpbench-tableformer/test/shard_000000_000000.parquet")
        # dataset.save_to_disk("benchmarks/dpbench-tableformer-dataset")

        test_path = str(ds_path / "test" / "*.parquet")
        train_path = str(ds_path / "train" / "*.parquet")

        test_files = glob.glob(test_path)
        train_files = glob.glob(train_path)
        logging.info(f"test-files: {test_files}, train-files: {train_files}")

        # Load all files into the `test`-`train` split
        ds = None
        if len(test_files) > 0 and len(train_files) > 0:
            ds = load_dataset(
                "parquet", data_files={"test": test_files, "train": train_files}
            )
        elif len(test_files) > 0 and len(train_files) == 0:
            ds = load_dataset("parquet", data_files={"test": test_files})

        logging.info(f"oveview of dataset: {ds}")

        table_evaluations = []
        # ds = datasets.load_from_disk(ds_path)
        if ds is not None:
            ds_selection: Dataset = ds[split]

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Table evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            # gt_doc_dict = data["GroundTruthDoclingDocument"]
            gt_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            gt_doc = DoclingDocument.model_validate_json(gt_doc_dict)
            # pred_doc_dict = data["PredictedDoclingDocument"]
            pred_doc_dict = data[BenchMarkColumns.PREDICTION]
            pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

            results = self._evaluate_tables_in_documents(
                doc_id=data[BenchMarkColumns.DOC_ID], true_doc=gt_doc, pred_doc=pred_doc
            )

            table_evaluations.extend(results)

        # Compute TED statistics for the entire dataset
        teds_simple = []
        teds_complex = []
        teds_all = []
        for te in table_evaluations:
            teds_all.append(te.TEDS)

            if te.is_complex:
                teds_complex.append(te.TEDS)
            else:
                teds_simple.append(te.TEDS)

        dataset_evaluation = DatasetTableEvaluation(
            evaluations=table_evaluations,
            TEDS=compute_stats(teds_all),
            TEDS_simple=compute_stats(teds_simple),
            TEDS_complex=compute_stats(teds_complex),
        )
        return dataset_evaluation

    def _evaluate_tables_in_documents(
        self,
        doc_id: str,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        structure_only: bool = False,
    ) -> list[TableEvaluation]:
        r""" """
        table_evaluations = []
        true_tables = true_doc.tables
        pred_tables = pred_doc.tables

        # logging.info(f"#-true-tables: {len(true_tables)}, #-pred-tables: {len(pred_tables)}")
        assert len(true_tables) == len(
            pred_tables
        ), "len(true_tables)!=len(pred_tables)"

        for table_id in range(len(true_tables)):  # , len(pred_tables)):

            try:
                true_table = true_tables[table_id]
                pred_table = pred_tables[table_id]

                is_complex = is_complex_table(true_table)

                true_html = true_table.export_to_html()
                pred_html = pred_table.export_to_html()

                # Filter out tags that may be present in GT but not in prediction to avoid penalty
                for stopword in self._stopwords:
                    predicted_html = pred_html.replace(stopword, "")
                for stopword in self._stopwords:
                    true_html = true_html.replace(stopword, "")

                true_html_obj = html.fromstring(true_html)
                pred_html_obj = html.fromstring(pred_html)

                teds = self._teds_scorer(true_html_obj, pred_html_obj, structure_only)
                # logging.info(f"teds: {teds}")

                teds = round(teds, 3)
                table_evaluation = TableEvaluation(
                    TEDS=teds,
                    is_complex=is_complex,
                    filename=doc_id,
                    table_id=table_id,
                    true_ncols=true_table.data.num_cols,
                    pred_ncols=pred_table.data.num_cols,
                    true_nrows=true_table.data.num_rows,
                    pred_nrows=pred_table.data.num_rows,
                )
                table_evaluations.append(table_evaluation)
            except Exception as exc:
                logging.error(
                    f"Table {table_id} from document {doc_id} could not be compared!"
                )

        return table_evaluations
