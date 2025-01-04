import os
import logging

from pathlib import Path

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.fintabnet.create import (
    create_fintabnet_tableformer_dataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)



def main():

    odir = Path("./benchmarks/fintabnet-dataset")

    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_tab]:
        os.makedirs(_, exist_ok=True)
    
    create_fintabnet_tableformer_dataset(
        output_dir=odir_tab
    )
    

    
if __name__ == "__main__":
    main()    
