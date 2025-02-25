# FinTabNet Benchmarks

Create and evaluate FinTabNet dataset using a single command. This command downloads from Huggingface the [FinTabNet_OTSL dataset](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL)) and runs the evaluations for tableformer using the first 1000 samples.

```sh
poetry run python docs/examples/benchmark_tableformer_fintabnet.py
```

## Layout Evaluation

Create the evaluation report:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b FinTabNet \
    -i benchmarks/FinTabNet-dataset/tableformer \
    -o benchmarks/FinTabNet-dataset/tableformer
```

[Tableformer evaluation json](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b FinTabNet \
    -i benchmarks/FinTabNet-dataset/tableformer \
    -o benchmarks/FinTabNet-dataset/tableformer
```

![TEDS plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer-delta_row_col.png)

![TEDS struct only plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.png)

[TEDS struct only report](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.txt)
