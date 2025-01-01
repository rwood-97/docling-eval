import json
import logging
import os
import subprocess
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.dpbench.create import (
    create_dpbench_layout_dataset,
    create_dpbench_tableformer_dataset,
)
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)
from docling_eval.utils.repository import clone_repository, is_git_lfs_installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    REPO_URL = "https://huggingface.co/datasets/upstage/dp-bench.git"  # Replace with your repo URL

    idir = Path("./benchmarks/dp-bench-original")

    if is_git_lfs_installed():
        clone_repository(REPO_URL, idir)
    else:
        logger.error("Please install Git LFS and try again.")

    odir = Path("./benchmarks/dp-bench-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 1.0

    if True:
        create_dpbench_layout_dataset(
            dpbench_dir=idir, output_dir=odir_lay, image_scale=image_scale
        )

        create_dpbench_tableformer_dataset(
            dpbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
        )

    if True:
        save_fn = (
            odir
            / f"evaluation_{BenchMarkNames.DPBENCH.value}_{EvaluationModality.LAYOUT.value}.json"
        )

        layout_evaluator = LayoutEvaluator()
        layout_evaluation = layout_evaluator(odir_lay, split="test")

        logging.info(f"writing results to {save_fn}")
        with open(save_fn, "w") as fd:
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

        data, headers = layout_evaluation.to_table()
        logging.info(
            "Class mAP[0.5:0.95] table: \n\n"
            + tabulate(data, headers=headers, tablefmt="github")
        )

    if True:
        save_fn = (
            odir
            / f"evaluation_{BenchMarkNames.DPBENCH.value}_{EvaluationModality.TABLEFORMER.value}.json"
        )

        figname = (
            odir
            / f"evaluation_{BenchMarkNames.DPBENCH.value}_{EvaluationModality.TABLEFORMER.value}.png"
        )

        table_evaluator = TableEvaluator()
        table_evaluation = table_evaluator(odir_tab, split="test")

        logging.info(f"writing results to {save_fn}")
        with open(save_fn, "w") as fd:
            json.dump(table_evaluation.model_dump(), fd, indent=2, sort_keys=True)

        data, headers = table_evaluation.TEDS.to_table()
        logging.info(
            "TEDS table: \n\n" + tabulate(data, headers=headers, tablefmt="github")
        )

        table_evaluation.TEDS.save_histogram(figname=figname)


if __name__ == "__main__":
    main()
