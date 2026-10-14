# Release traceability: v1.0.1 to v2.0.0

The public tag `v1.0.1` points to
`be6b8cecbba5de28d30591ecca9fabbd17a35f9f` (`update files`). Its concept DOI
is <https://doi.org/10.5281/zenodo.21830689>. It contained the 342-image VIA
copy and manifest, the 12,656-configuration fixed-feature study, selected
ten-fold/LOOCV outputs, manuscript tables, code, and corrected execution
records. Its historical CITATION.cff still declared version 1.0.0; the tag
identifies the v1.0.1 tree. All 2,681 files in that tree remain in v2.0.0.

## Additions for the revised article

| Directory | Added evidence |
| --- | --- |
| `external-data/` | Retained 1,576-image external release, evaluated/raw/released manifests, exclusions, frozen labels/scene groups and closed blind-review records |
| `results/fine_tuning_recovered/` | Matched fine-tuning, five CNNs and three initialization seeds; out-of-fold predictions, recovery provenance, comparisons and cost tables |
| `results/external_test/` | VIA-only frozen model/adapter/threshold records, external predictions, grouped uncertainty and sensitivity analyses |
| `results/learning_curve/` | Scene-disjoint external adaptation splits, predictions, metrics and measured update costs |
| `results/stage1_verification/`, `docs/STAGE1_NESTED_EXTERNAL_REPORT.md` | Preregistered nested/external protocol, diagnostic gate checks and planning estimates |
| `results/stage2_verification/`, `docs/STAGE2_REPORT.md` | Stage-2 verification, frozen external evaluation and learning-curve completion checks; scientific results under external_test/ and learning_curve/ |
| `results/stage3/`, `docs/STAGE3_REPORT.md` | Post hoc grouped VIA validation, thresholds, domain shift, attributes, clusters, final author error categories and blind relabeling |
| `results/decision_attribution/` | Decision-attribution outputs and spatial explanations; the attribution script already existed in v1.0.1 |

## Changes to pre-existing files

README.md now uses the revised article title and describes external evaluation,
matched fine-tuning, privacy modifications, licenses and reproduction commands.
CITATION.cff updates the title/version and adds the existing concept DOI and
external-data license notice; authors are unchanged and its release date must
be confirmed at publication. docs/PROVENANCE.md records the later protocols,
execution, recovery, author review and explicit AI-assisted draft provenance.
The complete legacy-file changes are:

| Existing v1.0.1 file | Reason for change |
| --- | --- |
| README.md | Revised title/scope, dataset/privacy/license documentation, reproduction and model assets |
| CITATION.cff | Revised title, version 2.0.0, existing concept DOI and data-license notice; authors retained |
| docs/PROVENANCE.md | Later protocols/execution, recovery and final author-review provenance |
| requirements-experiments.txt | Add the plotting and pytest versions used by the later analyses |
| requirements-experiments-macos.txt | Record the same plotting/test dependencies for macOS |
| requirements-results.txt | Add scikit-learn, plotting and pytest dependencies for results reproduction |
| scripts/build_manuscript_tables.py | Reuse src/selection.py so original ranking/selection rules are shared with later studies |
| src/artifact.py | Factor common binary prediction metrics into prediction_metrics, including obstructed false-negative rate |
| .gitignore | Ignore separately installed final model/encoder/adapter binaries; this change is new in E′ |

The complete name/status comparison accompanies the private author review as
`diff_v1.0.1_v2.0.0.txt`; it contains no deletions.

No v1.0.1 file was removed. Its checkpoint migration archive, cache-status
records and workstation workflow are retained for historical traceability.
`corrected_results/smoke_optimized/per_sample_predictions/group_0000.csv.gz`
was initially omitted as an optimized-smoke diagnostic outside the Phase-C
scientific allowlist; it is restored byte-for-byte from v1.0.1 and must remain.
It was not substituted by another file. It is a historical diagnostic shard,
not a source of manuscript metrics.
Only later-added internal assistant notes, AI draft suggestions and the local
stage-2 publishing helper were removed; final author records are unchanged.

## Revised article to artifact

| Revised section/table | Script | Result file |
| --- | --- | --- |
| Tables 1–2: architectures/search space | `src/artifact.py`; `scripts/generate_full_search_config.py` | `MODEL_SPECS`; `configs/full_search.json` |
| Tables 3–6: approaches A–D | `scripts/build_manuscript_tables.py` | `results/full_search/manuscript_tables/approach_a_ranking.csv` through `approach_d_ranking.csv`, corresponding `_rows.tex` |
| Table 7: finalists | `scripts/build_manuscript_tables.py` | `results/full_search/manuscript_tables/selected_approaches.csv` |
| Table 8: ten-fold/LOOCV | `scripts/analyze_article_followup.py` | `results/article_followup/validation_protocol_comparison.csv` |
| Table 9: matched fine-tuning | `scripts/analyze_fine_tuning.py` | `results/fine_tuning_recovered/manuscript_tables/fold_metric_summary.csv`; `initialization_seed_summary.csv` |
| Table 10: external generalization | `scripts/analyze_external_test.py` | `results/external_test/manuscript_tables/seed_mean_metrics.csv`; `cluster_bootstrap_wilson.csv` |
| Table 11: error categories | `scripts/stage3_error_analysis.py`; `scripts/build_stage3_report.py --mode final` | `results/stage3/error_analysis/manuscript_tables/error_category_counts.csv` |
| Table 12: computational costs | `scripts/analyze_fine_tuning.py` | `results/fine_tuning_recovered/manuscript_tables/computational_cost_summary.csv`; `training_cost.csv` |
| Learning-curve section | `src/external_test.py` (`learning_curve`) | `results/learning_curve/manuscript_tables/` |
| Spatial explanation section | `scripts/build_decision_attribution.py` | `results/decision_attribution/` |

## Distribution boundaries and hashes

Trained model and probability-adapter binaries are supplied separately as
GitHub v2.0.0 release attachments: [MODELS.md](../MODELS.md) documents packages,
licenses and `scripts/fetch_models.py` authentication/extraction. They are not
inside Git or the Zenodo source-code archive. Feature caches are omitted.
Original paths and SHA-256 records remain in frozen model and
execution provenance, including `results/external_test/final_models.json`
and `results/stage3/protected_before.json` / `protected_after.json`.
The 353 excluded external images and duplicate blind-package image copies are
omitted; raw/clean manifests and the exclusion audit retain their identifiers
and original hashes, and `external-data/blind_relabel/key.csv` retains the
blind-to-source mapping. Private privacy galleries and review ZIPs remain
private. `external-data/external_release_manifest.csv` binds all 1,576 evaluated
and released hashes: 20 local Gaussian-blur regions (19 plates, one face) in
16 images, with 1,560 intact. Intact scores were identical and maximum accuracy
variation per model/seed was 0.13 percentage points. Published article scores
used original evaluated images before anonymization.

Software is MIT; released external images, labels/groups/annotations are
CC BY 4.0. VIA retains its original Breve & Fischer attribution. The concept
DOI is unchanged; a new version DOI is recorded only after publication.
