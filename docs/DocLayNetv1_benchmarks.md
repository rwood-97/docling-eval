# DocLayNet V1 Benchmarks

Create and evaluate DocLayNetv1.2 dataset using a single command. This command downloads from Huggingface the [DocLayNetv1.2_OTSL](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2) and runs the evaluations using the PDF Docling converter for all supported modalities.

```sh
poetry run python docs/examples/benchmark_doclaynet_v1.py
```


## Layout Evaluation

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m layout \
    -b DocLayNetV1 \
    -i benchmarks/DocLayNetV1-dataset/layout \
    -o benchmarks/DocLayNetV1-dataset/layout
```

[Layout evaluation json](evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m layout \
    -b DocLayNetV1 \
    -i benchmarks/DocLayNetV1-dataset/layout \
    -o benchmarks/DocLayNetV1-dataset/layout
```

[mAP[0.5:0.95] report](evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout_mAP_0.5_0.95.txt)

![mAP[0.5:0.95] plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout_mAP_0.5_0.95.png)


## Markdown text Evaluation

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m markdown_text \
    -b DocLayNetV1 \
    -i benchmarks/DocLayNetV1-dataset/layout \
    -o benchmarks/DocLayNetV1-dataset/layout
```

[Markdown text json](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text.json)


Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m markdown_text \
    -b DocLayNetV1 \
    -i benchmarks/DocLayNetV1-dataset/layout \
    -o benchmarks/DocLayNetV1-dataset/layout
```

[Markdown text report](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text.txt)


![BLEU plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_BLEU.png)

![Edit distance plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_edit_distance.png)

![F1 plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_F1.png)

![Meteor plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_meteor.png)

![Precision plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_precision.png)

![Recall plot](evaluations/DocLayNetV1/evaluation_DocLayNetV1_markdown_text_recall.png)
