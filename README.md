# Cross-Analysis of Fixed CNN Feature Extractors for Path-Obstruction Classification to Assist Visually Impaired People

[![DOI](https://zenodo.org/badge/1297033488.svg)](https://doi.org/10.5281/zenodo.21830689)

Reproducibility artifact for the study by **Jefferson Antonio Ribeiro Passerini
and Fabricio Aparecido Breve**, São Paulo State University (UNESP), Rio Claro,
Brazil. The v2.0.0 release tree contains the software, frozen predictions and
analyses, the original VIA convenience copy, and the anonymized external set.
For the version history and revised-article mapping, see
[release traceability](docs/RELEASE_TRACEABILITY.md).

The experiment compares ImageNet-pretrained CNNs used as fixed feature
extractors with matched fine-tuning for classifying pedestrian paths as clear
(label 1) or obstructed (label 0). VIA has 342 images (175 clear, 167 obstructed);
the geographically and temporally independent external test has 1,576 images
(844 clear, 732 obstructed). CNNs, downstream classifiers, probability adapters
and operating thresholds were selected or trained using VIA alone before the
external evaluation.

## Fixed-feature study

The complete search comprises 12,656 configurations, 1,582 preprocessing groups
and ten stratified VIA folds with seed 1980. PCA/UMAP/Relief-F, scaling and
classifier fitting use only each training partition. The four approaches are:

| Approach | Feature construction |
| --- | --- |
| A | Full single-CNN or concatenated feature vector |
| B | Reduce/select one CNN's feature vector |
| C | Concatenate two CNNs, then reduce/select jointly |
| D | Reduce/select each CNN, then concatenate |

The frozen search and finalists are under `results/full_search/` and
`configs/article_selected_models_10fold.json`. Selected ten-fold and LOOCV
predictions are under `results/article_selected_10fold/` and
`results/article_selected_loocv/`; their stability comparison is under
`results/article_followup/`. Selected-finalist VIA performance is an internal,
exploratory benchmark that inherits selection optimism. The geographic external
test is the main generalization analysis. Protocol: `docs/EXTERNAL_TEST_PROTOCOL.md`.

## External test set

`external-data/via-dataset-extended/` contains **only the 1,576 retained release
images** collected by smartphone in Fernandópolis and Jales, SP, in 2024–2025,
with 1,378 scene groups. Cleaning removed 353 of the 1,929 raw files, including
350 exact matches to VIA and three internal duplicates. Historical manifests, exclusion audits, reviewed labels and scene metadata are under `external-data/`.

Local automatic face/plate screening was reviewed manually by the author.
Local Gaussian blur was applied to **20 approved regions (19 plates and one
face) in 16 images**. The other 1,560 images are byte-identical to their originals
and reproduced exactly identical frozen-model scores. The maximum absolute
accuracy change per model and seed was **0.13 percentage points**.
The published article results were obtained from the original evaluated images,
before anonymization. Frozen predictions and tables remain unchanged.
`external-data/external_release_manifest.csv` contains evaluated/released hashes
and the anonymization record. Collection, definitions, privacy and citation
details: `external-data/via-dataset-extended/README.md`.

`external-data/blind_relabel/` distributes the instructions, key and author
responses after the closed analysis. Duplicate blind image files, private
privacy-review galleries and the internal release-check directory are excluded.

## Matched fine-tuning

Five architectures (ResNet101V2, EfficientNetB3, MobileNet, ResNet50 and
EfficientNetB4), with initialization seeds 1980, 1981 and 1982, use the same
recorded ten outer VIA folds as the fixed-feature study. Early stopping uses
an internal training-only validation split. Configuration D is specified in
`configs/fine_tuning.json`. Complete results and recovery provenance are under
`results/fine_tuning_recovered/`; the earlier partial run is retained under
`results/fine_tuning/`.

Primary fine-tuning performance is the mean of individual-seed metrics; the
mean-probability ensemble is separately labeled as sensitivity. Published
comparisons include paired bootstrap intervals, Wilcoxon/Pratt with Holm,
McNemar and the prespecified 3-point substitution margin. Final VIA-only models,
frozen thresholds and external predictions are documented under
`results/external_test/`. Model binaries are distributed separately as GitHub release attachments;
feature caches are omitted. Their hashes and measured results are retained. Historical numerical analyses
can be regenerated from the supplied prediction CSVs. Exact rerunning of the
original-image inference requires the evaluated inputs and frozen binaries;
the released privacy copies intentionally have different hashes for 16 images.

## Trained models

The seven model-asset packages contain the exact final frozen pipelines,
encoders, fine-tuned models and probability adapters used for external testing.
See [MODELS.md](MODELS.md) for sizes, hashes, licenses, download URLs and
`python scripts/fetch_models.py` verification/extraction instructions.
These GitHub release attachments are outside Git and are not part of the
Zenodo source-code archive.

## Stage-3 analyses

`results/stage3/` contains exploratory/post hoc analyses of the frozen results:
grouped VIA validation, threshold/ROC analysis, domain shift, image attributes,
12 descriptive clusters, author categories for 148 consensus errors, low-light
subsets and the closed blind relabeling of 100 external images. Author labels,
categories and cluster names are retained exactly. No post hoc correction
changes a frozen prediction or label. Summary: `docs/STAGE3_REPORT.md`.
`docs/PROVENANCE.md` records prespecification, execution and approved deviations.
The final visual scene-group review was performed by the author
(`results/scene_groups/author_review_application.json`). AI-assisted drafts of
cluster names and error categories were reviewed, corrected and finalized by
the author on 2026-10-13; only the final author records are distributed.

## Learning curve

`results/learning_curve/` holds the secondary experiment using 0%, 25%, 50% and
100% external training subsets with recorded scene-disjoint test splits. These
are adaptation experiments, separate from the main one-shot external test.
Metrics and measured update costs are under
`results/learning_curve/manuscript_tables/`. Protocol and execution summary:
`docs/STAGE2_REPORT.md` and `configs/external_test.json`.

## Reproducing each manuscript table

The table numbering follows the revised manuscript. Supplied CSVs are the
numerical sources; table generation does not require retraining models.

| Article table | Script/source | Output file(s) |
| --- | --- | --- |
| 1: CNN architectures | `src/artifact.py` (`MODEL_SPECS`) | Architecture definitions and published parameter specifications |
| 2: search space | `scripts/generate_full_search_config.py` | `configs/full_search.json` (frozen enumeration) |
| 3: Approach A | `scripts/build_manuscript_tables.py` | `results/full_search/manuscript_tables/approach_a_ranking.csv`, `approach_a_rows.tex` |
| 4: Approach B | Same script | `results/full_search/manuscript_tables/approach_b_ranking.csv`, `approach_b_rows.tex` |
| 5: Approach C | Same script | `results/full_search/manuscript_tables/approach_c_ranking.csv`, `approach_c_rows.tex` |
| 6: Approach D | Same script | `results/full_search/manuscript_tables/approach_d_ranking.csv`, `approach_d_rows.tex` |
| 7: finalist comparison | Same script | `results/full_search/manuscript_tables/selected_approaches.csv`, `selected_approaches_per_fold.csv` |
| 8: ten-fold/LOOCV stability | `scripts/analyze_article_followup.py` | `results/article_followup/validation_protocol_comparison.csv`, `validation_protocol_rows.tex` |
| 9: matched fine-tuning on VIA | `scripts/analyze_fine_tuning.py` | `results/fine_tuning_recovered/manuscript_tables/fold_metric_summary.csv`, `initialization_seed_summary.csv` |
| 10: external test | `scripts/analyze_external_test.py` | `results/external_test/manuscript_tables/seed_mean_metrics.csv`, `cluster_bootstrap_wilson.csv` |
| 11: consensus-error categories | `scripts/stage3_error_analysis.py` | `results/stage3/error_analysis/manuscript_tables/error_category_counts.csv`, `consensus_errors.csv` |
| 12: computational costs | `scripts/analyze_fine_tuning.py` | `results/fine_tuning_recovered/manuscript_tables/computational_cost_summary.csv`, `training_cost.csv` |

For Table 9, the comparison with the previously published Breve result is
literature data cited by the manuscript, not a newly generated experiment.
Further ROC, grouped-validation, domain-shift and cluster analyses come from
`scripts/analyze_stage3.py`; the stage-three report is generated by
`scripts/build_stage3_report.py --mode final`. Spatial attribution figures use
`scripts/build_decision_attribution.py` and `results/decision_attribution/`.

Examples for regenerating the numerical table sources:

```bash
python scripts/build_manuscript_tables.py --results results/full_search --config configs/full_search.json
python scripts/analyze_article_followup.py --tenfold-results results/article_selected_10fold --loocv-results results/article_selected_loocv --output results/article_followup
python scripts/analyze_fine_tuning.py --results results/fine_tuning_recovered --config configs/fine_tuning.json --bootstrap-repetitions 10000 --seed 1980
python scripts/analyze_external_test.py --results results/external_test
```

Reproduce numerical analyses with the commands above. Legacy workstation
launchers and monitoring notes remain for v1.0.1 traceability; the later local
stage-2 publishing helper is omitted.

The legacy default `configs/selected_models.json` in `src/run_experiments.py`
is absent. Supply an existing config explicitly, such as
`--config configs/full_search.json` for the full VIA search or
`--config configs/article_selected_models_10fold.json` for the four finalists.

## Verifying data integrity

Run from the repository root:

```bash
python scripts/verify_dataset.py
python scripts/verify_external_release.py
python -m pytest -q
```

The original VIA verifier uses `data/dataset_manifest.csv`. The release verifier
authenticates the release manifest and checks every released SHA-256, all
1,576 filenames, the 844/732 classes, 1,378 groups, 16 anonymized images and
the absence of the 353 excluded filenames. Historical raw/evaluated manifests
refer to the evaluated inputs; `external_release_manifest.csv` is the integrity
reference for the distributed external JPEG files. The historical cleaning
test reads the raw manifest rather than expecting raw images in this tree.
GPU integration tests are optional as declared in `tests/conftest.py`.

Dependency snapshots: `requirements-results.txt`, `requirements-experiments.txt`,
`requirements-scene-groups-core.txt`, `requirements-scene-groups.txt` and
`environment.yml`. Requirements for macOS are recorded separately.

## Data, licenses and citation

Software is **MIT** (`LICENSE`). The released external images, labels, scene
groups and annotations are **CC BY 4.0** (`LICENSE-DATA`), with the privacy
modifications described above. Retain attribution and identify further changes
when redistributing the external data.

The VIA copy under `via-dataset/` belongs to its original authors,
**Breve & Fischer (IJCNN 2020)**, and is included for convenience. Its canonical
source is <https://github.com/fbreve/via-dataset>; retain the original credit
and provenance of that dataset. `LICENSE-DATA` covers the external collection,
not the original VIA data.

Use `CITATION.cff` for the software/artifact citation. The concept DOI is
<https://doi.org/10.5281/zenodo.21830689>. The `date-released` field in
`CITATION.cff` will be confirmed or updated on the publication date in Phase F.
The new Zenodo version DOI is assigned after publication.
