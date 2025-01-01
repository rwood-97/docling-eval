# Docling-eval


[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ds4sd.github.io/docling/)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)

Evaluate [Docling](https://github.com/DS4SD/docling) on various datasets.

## Features

Evaluate docling on various datasets. You can use the cli

```sh
docling-eval % poetry run evaluate --help
2024-12-20 10:51:57,593 - INFO - PyTorch version 2.5.1 available.

 Usage: evaluate [OPTIONS]

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --task        -t      [create|evaluate|visualize]                                                                Evaluation task [default: None] [required]                                                                              │
│ *  --modality    -m      [end-to-end|layout|tableformer|codeformer]                                                 Evaluation modality [default: None] [required]                                                                          │
│ *  --benchmark   -b      [DPBench|OmniDcoBench|WordScape|PubLayNet|DocLayNet|Pub1M|PubTabNet|FinTabNet|WikiTabNet]  Benchmark name [default: None] [required]                                                                               │
│ *  --input-dir   -i      PATH                                                                                       Input directory [default: None] [required]                                                                              │
│ *  --output-dir  -o      PATH                                                                                       Output directory [default: None] [required]                                                                             │
│    --help                                                                                                           Show this message and exit.                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## End to End examples

### DP-Bench

Using a single command,

```sh
poetry run python ./docs/examples/benchmark_dpbench.py
```

<details>
<summary><b>Layout evaluation for DP-Bench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m layout -b DPBench -i <location-of-dpbench> -o ./benchmarks/dpbench-layout
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m layout -b DPBench -i ./benchmarks/dpbench-layout -o ./benchmarks/dpbench-layout
```

| id |          label | MaP[0.5:0.95] |
| -- | -------------- | ------------- |
|  0 |    page_header |         0.151 |
|  1 |           text |         0.678 |
|  2 | section_header |         0.443 |
|  3 |       footnote |         0.221 |
|  4 |        picture |         0.761 |
|  5 |        caption |         0.458 |
|  6 |    page_footer |         0.344 |
|  7 | document_index |         0.755 |
|  8 |        formula |         0.066 |
|  9 |          table |         0.891 |
</details>

<details>
<summary><b>Table evaluations for DP-Bench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m tableformer -b DPBench -i <location-of-dpbench> -o ./benchmarks/dpbench-tableformer
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b DPBench -i ./benchmarks/dpbench-tableformer -o ./benchmarks/dpbench-tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b DPBench -i ./benchmarks/dpbench-tableformer -o ./benchmarks/dpbench-tableformer
```

The final result can be visualised as,

![DPBench_TEDS](./docs/evaluations/evaluation_DPBench_tableformer.png)
</details>

### OmniDocBench

Using a single command,

```sh
poetry run python ./docs/examples/benchmark_omnidocbench.py
```

<details>
<summary><b>Table evaluations for OmniDocBench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m tableformer -b OmniDocBench -i <location-of-omnidocbench> -o ./benchmarks/omnidocbench-tableformer
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b OmniDocBench -i ./benchmarks/omnidocbench-tableformer -o ./benchmarks/omnidocbench-tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b OmniDocBench -i ./benchmarks/OmniDocBench-dataset/tableformer/ -o ./benchmarks/OmniDocBench-dataset/tableformer/
```

The final result can be visualised as,

<table>
  <tr>
    <td>
      <img src="./docs/evaluations/evaluation_OmniDocBench_tableformer.png" alt="OmniDocBench_TEDS" width="400">
    </td>
    <td>
      <table>
        <thead>
          <tr>
            <th>index</th>
            <th>x0&lt;TEDS</th>
            <th>TEDS&lt;x1</th>
            <th>count</th>
            <th>%</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>00</td><td>0</td><td>0.05</td><td>3</td><td>0.909</td></tr>
          <tr><td>01</td><td>0.05</td><td>0.1</td><td>2</td><td>0.606</td></tr>
          <tr><td>02</td><td>0.1</td><td>0.15</td><td>14</td><td>4.242</td></tr>
          <tr><td>03</td><td>0.15</td><td>0.2</td><td>11</td><td>3.333</td></tr>
          <tr><td>04</td><td>0.2</td><td>0.25</td><td>7</td><td>2.121</td></tr>
          <tr><td>05</td><td>0.25</td><td>0.3</td><td>7</td><td>2.121</td></tr>
          <tr><td>06</td><td>0.3</td><td>0.35</td><td>8</td><td>2.424</td></tr>
          <tr><td>07</td><td>0.35</td><td>0.4</td><td>9</td><td>2.727</td></tr>
          <tr><td>08</td><td>0.4</td><td>0.45</td><td>5</td><td>1.515</td></tr>
          <tr><td>09</td><td>0.45</td><td>0.5</td><td>9</td><td>2.727</td></tr>
          <tr><td>10</td><td>0.5</td><td>0.55</td><td>9</td><td>2.727</td></tr>
          <tr><td>11</td><td>0.55</td><td>0.6</td><td>16</td><td>4.848</td></tr>
          <tr><td>12</td><td>0.6</td><td>0.65</td><td>7</td><td>2.121</td></tr>
          <tr><td>13</td><td>0.65</td><td>0.7</td><td>12</td><td>3.636</td></tr>
          <tr><td>14</td><td>0.7</td><td>0.75</td><td>31</td><td>9.394</td></tr>
          <tr><td>15</td><td>0.75</td><td>0.8</td><td>24</td><td>7.273</td></tr>
          <tr><td>16</td><td>0.8</td><td>0.85</td><td>42</td><td>12.727</td></tr>
          <tr><td>17</td><td>0.85</td><td>0.9</td><td>40</td><td>12.121</td></tr>
          <tr><td>18</td><td>0.9</td><td>0.95</td><td>48</td><td>14.545</td></tr>
          <tr><td>19</td><td>0.95</td><td>1</td><td>26</td><td>7.879</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>
</details>


## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.

## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
