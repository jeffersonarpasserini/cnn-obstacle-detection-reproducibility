# Artifact provenance

## Dataset

- Dataset: VIA path-obstruction image dataset
- Canonical source: <https://github.com/fbreve/via-dataset>
- Included sample count: 342 images
- Class counts: 175 clear and 167 obstructed
- Local integrity record: `data/dataset_manifest.csv`

## Experiment definition

- Random seed: 1980
- Validation: stratified 10-fold cross-validation with shuffling
- Complete configuration: `configs/full_search.json`
- Completed-run metadata: `results/full_search/run_metadata.json`
- Exact sample/fold mapping: `results/full_search/fold_assignments.csv`

Every learned preprocessing operation and classifier is fitted only on the
training indices supplied for its fold. ImageNet-pretrained CNN weights remain
fixed and serve only as feature extractors.

## Numerical sources for the manuscript

- Per-fold metrics: `results/full_search/per_fold_metrics.csv`
- Descriptive summaries: `results/full_search/summary_metrics.csv`
- Analysis code: `scripts/build_manuscript_tables.py`
- Generated article tables: `results/full_search/manuscript_tables/`

`analysis_manifest.json` records SHA-256 hashes of the full-search metrics and
configuration used to generate the article tables.

When this repository is released, record the release tag, commit identifier,
and archival DOI below:

- Release tag: pending
- Commit: pending
- DOI: pending
