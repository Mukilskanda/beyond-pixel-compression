# Data

Datasets are not redistributed in this repository. The loaders in `src/data/`
stream them from the HuggingFace Hub on first use and cache images here
(`data/*_cache/`, gitignored):

| Dataset | Source | Loader |
|---|---|---|
| DocVQA (validation) | `lmms-lab/DocVQA`, config `DocVQA` | `src/data/docvqa_loader.py` |
| ChartQA (test) | `lmms-lab/ChartQA` | `src/data/chartqa_loader.py` |

Every logged call in `results/results.sqlite` records its image path, whose file
name is the DocVQA `questionId` (or `chartqa_test_<index>`), so the exact samples
can be re-fetched. The paper used samples in loader order; a window "offset o"
in Table IV means samples o to o+3.

To download the samples used in the paper (fetches in loader order until every
image referenced in the logs is present):

    python data/download_datasets.py

Licensing: DocVQA and ChartQA are distributed by their authors under their own
terms (see the dataset cards on the HuggingFace Hub). Check them before reuse.
