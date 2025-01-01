import json
import logging
import os
import subprocess
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.utils.repository import is_git_lfs_installed, clone_repository

from docling_eval.benchmarks.omnidocbench.create import (
    create_omnidocbench_layout_dataset,
    create_omnidocbench_tableformer_dataset,
)
"""
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    REPO_URL = "https://huggingface.co/datasets/opendatalab/OmniDocBench"  # Replace with your repo URL

    idir = Path("./benchmarks/OmniDocBench-original")

    if is_git_lfs_installed():
        clone_repository(REPO_URL, idir)
    else:
        logger.error("Please install Git LFS and try again.")

    odir = Path("./benchmarks/OmniDocBench-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 1.0

    """
    create_omnidocbench_layout_dataset(
        omnidocbench_dir=idir, output_dir=odir_lay, image_scale=image_scale
    )
    """
    
    create_omnidocbench_tableformer_dataset(
        omnidocbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
    )
        

if __name__ == "__main__":
    main()        
