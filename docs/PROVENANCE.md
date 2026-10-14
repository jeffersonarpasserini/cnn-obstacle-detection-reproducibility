# Artifact provenance

## Dataset

- Dataset: VIA path-obstruction image dataset
- Canonical source: <https://github.com/fbreve/via-dataset>
- Included sample count: 342 images
- Class counts: 175 clear and 167 obstructed
- Local integrity record: `data/dataset_manifest.csv`

## Fixed-feature experiment definition

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

## Matched task-specific fine-tuning

- Implementation and effective protocol: `src/fine_tuning.py`,
  `configs/fine_tuning.json`; training recipes D and optional A follow the
  architecture/settings specified for Breve (ICCSA 2023).
- ImageNet backbone at its existing `MODEL_SPECS` native input size, no top,
  global average pooling, Dense(128, ReLU, He-uniform), Dense(2, softmax);
  no augmentation. D trains all layers with Adam; A freezes the backbone
  (including inference-mode batch normalization/dropout) and uses RMSprop.
- The immutable reference is `results/full_search/fold_assignments.csv`,
  not a newly generated split. The ordered dataset content must match the
  canonical run's SHA-256. Each outer train reserves a seed-1980 stratified
  20% validation set, shared by initialization seeds 1980/1981/1982.
  Fine-tuning fits on the remaining 80% (subject to integer rounding, Breve
  2023 protocol); fixed PCA/classifier pipelines use 100% of the outer train.
  `inner_validation_indices.csv` and per-job `split_indices.json` preserve it.
- Validation alone controls LR reduction, stopping and best-weight restoration.
  Held-out images do not select epochs, thresholds or hyperparameters;
  clear means P(clear)>0.5, with equal softmax scores assigned to obstructed.
- Outputs are separate under `results/fine_tuning/`: individual fold/seed
  metrics and image probabilities, arithmetic fold seed means, explicitly
  labeled mean-probability image ensembles, and measured per-job costs.
  `run_metadata.json` binds the effective JSON, source/fold/config hashes,
  image-content identity, dependency versions, TensorFlow/CUDA/cuDNN and GPU.
  Fresh seeded worker processes, optimizer checkpoints, callback/best-weight
  state, atomic manifests and checksum-validated completed artifacts support
  resumption. Uncommitted epochs restart; interrupted stochastic-backbone
  RNG state is not claimed to be bitwise identical to an uninterrupted run.
- Before training, `--prefetch-weights` downloads/checks all five configured
  no-top ImageNet weight files in the Keras model cache and records their
  SHA-256 hashes and source URLs in `run_metadata.json`. Training fails with
  a prefetch instruction if any cache entry is missing; each worker checks
  its recorded hashes before loading weights.
- Deployment models use atomic native `.keras` saves without optimizer state;
  optimizer checkpoints remain separate. Eight held-out images must preserve
  both softmax probabilities (neural) or native continuous classifier scores
  (fixed), within absolute tolerance 1e-5, and exactly preserve decisions.
  `.keras` files and fixed downstream pipelines are checksummed in the
  completed-job artifact manifest.
- `--smoke-all` uses all five configured architectures, outer fold 1, one epoch,
  seed 1980 and no fixed-feature jobs. Per-architecture `smoke_report.json`
  records epoch time, peak GPU/host memory, effective batch/OOM reductions,
  reload checks and determinism. `--compare-smoke` requires a fresh second
  directory and exact predictions under unchanged protocol/data/source/runtime.
  Deterministic-operation failures retain their architecture, available
  operation and traceback without disabling determinism. Two successful
  Linux/GPU runs and report approval are prerequisites for the full study;
  local CPU verification does not replace them.
- `tests/test_fine_tuning_gpu.py --run-gpu` automates weight prefetch and two
  fresh Linux/GPU five-architecture smokes. It requires measured GPU memory,
  CPU/GPU batch-one latency, eight-image reload equivalence, intact native
  deployment/checkpoint artifacts, and exact independent-run predictions.
  `--gpu-output-dir` preserves reports and logs in a new directory. GPU tests
  are opt-in; an explicitly requested GPU run fails rather than skipping when
  the required runtime is unavailable.
- `scripts/configure_tensorflow_gpu.py` exposes the pip-installed NVIDIA
  libraries and `ptxas` through links inside the active project virtualenv,
  following TensorFlow's documented GPU setup. It refuses conflicting files
  and writes a virtualenv-local activation script for CUDA library/compiler
  paths. GPU tests set those paths before launching TensorFlow subprocesses.
  System CUDA and the training protocol are unchanged.
- Substitution prespecification: authors fix `substitution_margin_pp` (default
  3.0 percentage points) and accuracy/obstructed-recall endpoints before the
  full study. Startup records the actual config-file SHA-256, absolute source
  path and immutable UTC timestamp. Analysis requires the unchanged config
  hash and matching recorded/effective margin and endpoints; changed settings
  cannot be retrospectively analyzed under the original prespecification.
- Cost comparisons refit the four configurations in
  `configs/article_selected_models_10fold.json` using the recorded outer
  training folds, validating predictions/metrics against the canonical run.
  Timing includes the original CNN forward, native preprocessing and, where
  applicable, the fitted PCA plus classifier, not cached-vector-only inference.
  CPU/GPU desktop batch-one timings run from JPEG to decision, with ten
  warmups and 100 synchronized repetitions; they do not represent smartphone
  latency. VRAM is the TensorFlow training-allocator peak; missing GPU data
  stay missing. Deployment sizes exclude optimizer checkpoints; fixed
  downstream parameter counts explicitly include fitted PCA and classifier scalars.
  Additional costs isolate synchronized forward-only CNN time. For a real
  cached PCA/classifier label update, the fixed pipeline first fits the N-1
  label set without timing, then times the full-N refit used for canonical
  test predictions. Neural adaptation is the measured prescribed complete
  fresh fit, not a warm-start or a second training run. N-1 -> N populations
  are explicit: whole outer train for fixed pipelines, fit train excluding
  validation for neural models. The same deterministic internal-fit image
  supplies the new label to both sides, never a test/validation image. Host memory
  is fresh-process high-water RSS at training end, including framework/model
  and checkpoint overhead. `training_cost.csv` uses within-fold seed means
  then medians, quartiles and IQR across the ten folds; component-only neural
  timings and unavailable GPU values remain missing.
- Analysis: `scripts/analyze_fine_tuning.py`, generated
  `results/fine_tuning/manuscript_tables/`. Canonical fold statistics come from
  `results/full_search/per_fold_metrics.csv`; complete archived image predictions
  come from `results/article_selected_10fold/` and must reproduce those metrics.
  Wilcoxon/Pratt uses fold means over seeds and Holm across all model/endpoint
  pairs. Exact McNemar uses the labeled mean-probability ensemble. Wilson
  intervals use 167 truly obstructed images per seed/ensemble. The seed-1980
  10,000-replicate image-stratified paired bootstrap averages per-seed statistics
  within each common image resample; seeds are not independent observations.
- The fixed finalists were selected on these same folds. This extension is
  an exploratory matched comparison, not a nested estimate of the full
  configuration-selection procedure and not independent external validation.
- `substitution_analysis.csv` and LaTeX rows report D=fixed minus mean
  single-seed fine-tuning, paired class-stratified image-bootstrap 95% intervals
  (10,000/1980) and ten-fold paired-t sensitivity (df=9). Accuracy/obstructed
  recall support requires lower CI > -Delta; FNR support requires upper CI
  < +Delta. FNR differences/intervals are exact negative recall transforms
  from the same resamples. Margin equality is inconclusive; proportions and
  percentage-point columns are distinct.
- `--nested-predictions` accepts complete canonical external image predictions
  for A/B/C/D/winner, with default and optional safety labels and per-fold
  selected configurations. It replaces the performance source, validates
  image/fold identities and hashes the input; it does not generate nested
  results. Current finalist costs remain explicitly current-finalist costs,
  never imputed for nested-selected procedures. Root substitution outputs
  record the chosen side; separate manuscript output directories can preserve
  both comparisons.
- Verification: Python 3.12 / TensorFlow 2.21 CPU MobileNet/ImageNet smoke,
  checkpoint/completed-job resumption, frozen RMSprop control construction,
  and one real MobileNet+ResNet50/PCA(300)/SVM finalist fold. Full-study GPU
  results and GPU timing are not claimed by this implementation entry.
  Addendum checks also exercise retrospective-margin rejection, jointly paired
  resampling, strict substitution boundaries/FNR reversal, and a real cached
  finalist update. At that stage, fifty pytest checks passed locally.
  Archived/nested analysis CLI paths were exercised on explicitly synthetic
  verification fixtures; those fixtures are not research study results.
- Linux/GPU verification (2026-10-04 to 2026-10-05, local dates): 64 fine-tuning/protocol
  tests passed, followed by the GPU integration test. Both real five-architecture
  smokes passed on the RTX 4070 Ti / TensorFlow 2.21.0; all 175 held-out
  probabilities/decisions matched exactly between independent runs. Native
  reload checks had zero maximum difference on eight images per architecture.
  ResNet101V2, EfficientNetB3, MobileNet and ResNet50 used batch 16;
  EfficientNetB4 recorded OOM at 16 and completed at batch 8 in both runs.
  Reports, logs and native artifacts are local under
  `results/fine_tuning_gpu_check_20261004/{smoke_all,smoke_all_repeat}/`.
  These one-fold/one-epoch checks are separate from full-study results.

## Image-only scene / near-duplicate grouping

The following V1 record is historical. Its original configuration, CSVs,
metadata, embeddings, scripts and PNGs are now preserved byte-for-byte in
`results/scene_groups_v1/`, verified by `archive_manifest.json`.
`data/surface_groups.csv` is its original 194-group assignment. Original live
paths mentioned below refer to the V1 execution, not the current V2 artifacts.

- Implementation: `scripts/build_scene_groups.py` and
  `scripts/scene_groups_review.py`; effective parameters and the original
  image-only calibration history: `configs/scene_groups.json`.
- Inputs: all 342 VIA images, bound by individual SHA-256 hashes and ordered
  inventory fingerprint
  `53a1857c7234ea771edda441e09bd0419fca1b2827a885c4d0785bcf881489a7`.
  No labels, classifier predictions, folds or performance metrics enter
  hashing, semantic extraction, threshold selection, graph construction or
  overrides. Class-bearing filenames are used only as identifiers/order;
  class diagnostics start after both group CSVs have been saved.
- Low-level features: ImageHash 4.3.2 pHash/dHash/wHash, 64 bits on original
  and horizontal mirror; minimum Hamming over all four orientation pairs.
  Only pHash is an edge criterion; dHash/wHash support review.
- Semantic encoder: [DINOv2 ViT-S/14](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056),
  revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, CLS pooling,
  384 dimensions, L2 normalization; `model.safetensors` SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
  RGB, bicubic short-edge resize to 256, center crop 224, ImageNet
  normalization; full processor configuration is recorded.
- Prespecified unavailable-DINO fallback: [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268),
  revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, L2-normalized
  512-dimensional projected CLS; `pytorch_model.bin` SHA-256
  `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
  Weights/revisions are checked and fallback reasons are retained; the real
  342-image study proposal used DINO, not CLIP.
- Additional dependencies: `requirements-scene-groups.txt` pins PyWavelets
  1.8.0, Torch 2.7.1, Transformers 4.51.3, Hugging Face Hub 0.30.2,
  OpenCV headless 4.11.0.86 and scikit-image 0.25.2.
  Actual numerical-library versions are in `run_metadata.json`, including
  local NumPy 2.5.3, pandas 3.0.6, Pillow 12.3.0 and scikit-learn 1.9.1.
- Graph: pHash ≤10 OR cosine ≥0.795 OR measured homography-RANSAC inliers ≥30;
  deterministic union-find, seed 1980, two CPU threads, batches of eight.
  Structural scoring is optional and disabled in the study proposal; its
  candidate policy, grayscale 256×341 SSIM, ORB matching and RANSAC parameters
  are fixed in the configuration. Missing structural scores are not zeros.
- Calibration: sorted nearest-neighbor diagonal knee with 0.70 lower bound;
  its 233-image component triggered the >15% guard. Cosine was increased by
  0.005 to 0.795, where the largest component is 49 (14.3%).
  Original threshold/size history remains in `config.calibration.guard_history`.
  Subsequent fixed-threshold runs have their own execution history. A changed
  image fingerprint or actual encoder/revision requires new label-free
  calibration, not reuse of another dataset/model's threshold.
- Assignments: `data/groups_auto.csv` and `data/groups.csv`, 342 unique
  filenames each, SHA-256
  `35658506bef151a8240ef5472c16242f1b30f14cb7b58e6c05daffe99c511365`
  for both, with no study overrides.
  **194 groups**; size→count: 1→154, 2→25, 3→5, 4→5, 5→1,
  9→1, 10→1, 30→1, 49→1.
- Post-freeze diagnostics: two mixed-class groups; ten
  `StratifiedGroupKFold` folds (shuffle/1980) contain both classes in every
  train/test partition, disjoint group sets and exactly-once test coverage.
  This feasibility check fits no classifier and does not tune grouping.
- Review artifacts: `threshold_review/` has six blind pairs per cosine band,
  36 total, plus the nearest-neighbor distribution plot; `review_sheet.csv`
  lists every group/member and internal cosine statistics; `group_review/`
  contains every image in paginated PNGs. Source, image, configuration,
  CSV/JSON/PNG hashes and effective preprocessing are bound in
  `results/scene_groups/run_metadata.json`.
- **Author validation remains pending.** Assistant inspection of all 36
  sampled pairs and all pages of the two largest groups is recorded separately.
  `scene_0002` mixes indoor tiles and outdoor paths (minimum cosine 0.1688);
  `scene_0024` spans multiple mosaic-pavement designs (minimum 0.5468).
  Neither the size guard nor appearance similarity establishes ground-truth
  location or capture sequence. Resolve possible transitive merges before
  study use; without EXIF/capture metadata, independent scenes are not proven.
- Manual corrections: optional `data/groups_overrides.csv` requires known,
  unique filenames, nonempty group IDs and either per-row `reason` or a
  documented `--override-reason`. Metadata records previous/new IDs,
  justification and override-file SHA-256; automatic `pairs.csv` remains the
  scored automatic graph, not invented manual edge evidence.
- Verification: full real DINO CPU inference/grouping on all 342 images,
  68.3 s initially and 27.4 s on a subsequent complete inference run; identical
  assignment hashes. All 373 exported automatic edges were independently
  reproduced from real image hashes and stored DINO embeddings. A real-DINO
  random-image smoke kept four distinct images separate and co-grouped
  exact/mirrored copies. Real optional ORB/RANSAC duplicate-image smoke produced
  1,323 inliers and SSIM=1.0. Real CLIP extraction, unavailable-DINO fallback,
  calibration rebinding, numeric justified overrides and checksum-validated
  embedding-cache reuse were exercised in isolated temporary CLI fixtures,
  not research results. The integrated pytest run passed **59 tests**
  (seven subtests); two existing SVC deprecation warnings remain.
- Approved boundary: `src/nested_cv.py` is absent. The user selected grouping
  and review only; no grouped nested-CV result, grouped/ungrouped metric
  difference or confidence interval is claimed or fabricated. Existing
  experiment-result directories are protected from grouping output.

### Scene grouping V2 (2026-10-05)

- Authorized policy: the consolidated decision and latest resolution in
  `../../PROMPT_GROUPS_V2.md` supersede the initial proposal. The resolution was
  supplied explicitly by the author and registered locally before implementation.
- Implementation: `build_scene_groups.py`, `scene_groups_graph.py`,
  `scene_groups_geometry.py` and `scene_groups_review.py`. Graph/review dependencies
  use an isolated `cache/scene-groups-venv/`; no package or source change is made
  to the fine-tuning environment. `requirements-scene-groups-core.txt` pins this
  CPU environment; optional encoder dependencies remain in the full requirements.
- Exact V1 DINO embeddings are reused after checksum, pixel inventory, encoder,
  revision, pinned weight digest and dimension checks. Original embedding-runtime
  versions/preprocessing are retained; current graph-library versions are separate.
  No encoder inference or GPU training is part of this V2 execution.
- Candidates: pHash ≤16 OR mutual k=10 cosine neighbors. Accepted automatic links:
  pHash ≤10 OR cosine ≥0.795 AND ≥30 ORB homography-RANSAC inliers AND inlier ratio
  ≥0.25 AND at least three quadrants in both images, each with ≥3 inliers.
  Correspondences pass Lowe ratio 0.75 in both directions and are reciprocal.
  Ratio denominator is the reciprocal valid-match count; quadrants are TL/TR/BL/BR
  in each grayscale 256×341 image. Noncandidate geometry is `not_evaluated`.
- Connected components preserve all accepted links. The 17-image limit is an
  alert only. `data/edge_rejections.csv` removes reviewed links with reasons,
  identified reviewers and ISO calendar review dates before
  components are recalculated. `data/groups_overrides.csv` merges entire components
  into existing target groups; it cannot split them. Final validation checks that
  no accepted edge crosses groups. Both audit-file hashes and records are retained.
- Initial result: 342 images, 58,311 exported pairs, 1,054 evaluated candidates,
  341 groups (340 singletons, one pair), one pHash link and no semantic-geometric
  links. Five candidates have ≥30 inliers; only one passes the complete geometric
  gate and it fails the fixed cosine threshold (0.6332). No thresholds were tuned,
  no manual links rejected/merged, and no size alert occurred. This result cannot
  rule out related captures that the automatic rules failed to identify.
- Review: one two-image mosaic; bridge correspondence mosaics are produced for
  groups larger than two (none in this execution). Five additional correspondence
  mosaics expose geometric candidates, including exclusions. Group statistics
  distinguish measured internal minima from unevaluated pairs. All review PNGs
  use hash-based IDs rather than class-bearing filenames.
- Freeze: group/pair hashes are recorded before label/fold diagnostics. Ten folds
  contain 34–36 images, 17–19 clear and 16–17 nonclear; disjoint groups and both
  classes are checked. Balance is diagnostic only. These generated folds do not
  replace `results/full_search/fold_assignments.csv`.
- Author scene validation remains pending. No grouped nested CV or additional
  fine-tuning comparison is executed. Any later grouped-fold fine-tuning comparison
  remains a planning decision after grouped/ungrouped nested-CV results exist.
- Historical snapshot-bound `review_recommendations.json` records preliminary
  visual proposals. The final visual review was performed by the author and is
  recorded in `results/scene_groups/author_review_application.json`; the final
  author decisions, rather than internal assistant notes, are distributed.
- Verification: 40 grouping tests and seven subtests passed, including reciprocal
  real ORB/RANSAC matching, both-image coverage, rejected-bridge recomputation,
  whole-component merging, immutable archives, complete CLI artifact hashes and
  correspondence mosaics. `final_validation.json` records checksum agreement,
  independently reproduced geometric/edge gates and diagnostic folds, and all
  246 archived V1 file checks. `protected_training_verification.json` confirms
  that seven protected file digests and all 69 training distribution versions
  match the pre-implementation snapshot. All supplementary report hashes are
  registered in `run_metadata.json`.

### Author scene-group review applied (2026-10-06)

- Authority: `../../PROMPT_GROUPS_APPLY_REVIEW.md` and the author-supplied
  `data/edge_rejections.csv` / `data/groups_overrides.csv` at repository commit
  `b0954eab6de7ae91a8f379aeaaf08d19ee55827c`. Both CSVs, their exact headers,
  reasons and destination IDs were verified and preserved byte for byte.
- One rejection removes the only confirmed pHash edge between `clear.020.jpg`
  and `nonclear.141.jpg`. The original `groups_unreviewed.csv` is unchanged
  (341 groups); reviewed `groups_auto.csv` has 342 singleton components.
- Four whole-component unions produce 338 final groups (334 singletons and
  four pairs): nonclear.102/104 → scene_0278; nonclear.116/117 → scene_0292;
  clear.132/nonclear.151 → scene_0133; nonclear.053/109 → scene_0229.
  Only scene_0133 mixes classes. The rejected pair and clear.174/nonclear.166
  remain separated. The largest group has two images; there are no size alerts
  or accepted links crossing groups.
- Execution used the exact requested CPU command and reused the pinned,
  checksum-validated V1 embeddings, with no encoder inference. The generator
  took 37.3 seconds, evaluated the same 1,054 candidates among 58,311 pairs,
  and changed no configuration, threshold or grouping source. Obsolete group
  mosaics were removed; only the four final pair mosaics remain.
- Audit: `author_review_application.json`, summary and run metadata record
  reviewer Jefferson Antonio Ribeiro Passerini (author), review date 2026-10-06,
  reasons, input hashes and affected/resulting group members. Reviewer/date for
  all four merge audits use the explicitly supplied shared author-review context;
  the three-column merge CSV was not extended. The generator's default pending
  status was annotated as approved only after verifying the supplied decisions.
  The original image-only freeze timestamp and group/pair hashes are preserved,
  and the freeze is independently confirmed to precede fold diagnostics.
- The groups are frozen for the planned grouped nested CV. Independent
  `StratifiedGroupKFold(10, shuffle=True, random_state=1980)` reproduction found
  34–36 test images per fold, 17–19 clear and 16–17 nonclear, disjoint train/test
  groups, both classes in every partition and exactly-once test coverage.
  Balance was reported only; no group or threshold was adjusted in response.
- Verification: 40 tests and seven subtests passed. `final_validation.json` and
  `protected_training_verification.json` approve the checks against a current
  pre-task baseline: 23 protected file hashes and all 69 training package
  versions are unchanged; 4,262 protected output files preserve their inventory,
  sizes, mtimes and inodes, with content hashes also checked for files up to
  8 MiB. All 246 V1 archived artifact hashes remain valid. The completed
  fine-tuning results, training code/environment and original full-search folds
  were not modified. No classifier or nested CV was executed.
- Snapshot-bound proposal CSV/JSON bytes remain historical records, unchanged.
  The final visual review was performed by the author
  (`results/scene_groups/author_review_application.json`). Author review covers
  the six inspected pairs and does not prove independence of all remaining
  image captures. Internal assistant review notes are omitted from the release.

### Fixed-feature refit recovery (2026-10-05)

- Original execution: `results/fine_tuning/` contains 150 completed neural jobs,
  two completed fixed-feature jobs and a failed ResNet101V2/logistic fold-3 job.
  All 762 artifact checksums of the 152 completed jobs were verified; recomputed
  neural metrics agree with the saved predictions. The original executor's
  SHA-256 is `713cebd89203f0757dece908ea3d8d2a0e7697c0176be963ca0a86a446f8a3c7`.
- Diagnosis: the canonical AUC is 0.9826388888888888; separate outer-training
  and test batches yielded 0.9861111111111112 with identical class decisions.
  Isolated eager/graph and batch-size comparisons identified batch membership
  as the cause. Globally ordered image-only extraction restores the reference
  metric without changing the classifier, its hyperparameters or metric code.
- Repair: `src/fixed_refit.py` extracts frozen ImageNet features in globally
  sorted filename order with the batch size recorded in the canonical command
  (default four). Cache identity binds dataset contents, weights, architecture,
  runtime, policy and helper source; feature-file checksums are checked on reuse.
  PCA/classifiers fit exclusively on outer-training indices. Every canonical
  metric is still checked at absolute tolerance 1e-8; no mismatch is suppressed.
- Cost scope: timed outer-training/test forwards are replayed independently of
  the canonical feature cache, then the actual downstream fit/prediction uses
  canonical activations. Global feature setup and extraction provenance are
  recorded separately in `canonical_features.json` and cost fields; setup is
  excluded from training time. These warmed-extractor measurements must disclose
  that setup separately. The N-1 to N cached-label-update protocol is unchanged.
- Recovery: `scripts/resume_fine_tuning.py` creates a separate, audited run in
  `results/fine_tuning_recovered/`. It requires original runtime/data/weights,
  unchanged configuration/folds/references, complete neural coverage and valid
  artifact checksums. AST comparison rejects changes to any neural/protocol code
  beyond the fixed-refit function and exact artifact/source-hash bookkeeping
  additions. Original byte contents are preserved; model artifacts are hard
  linked and child manifests record both identities and original manifest hashes.
  All 150 neural jobs are reused; all 40 fixed benchmarks are scheduled anew.
- Automated verification: 68 tests passed, including 14 cache/recovery tests.
  Tests cover canonical batch composition, checksum failures, protocol changes,
  changed or corrupted recovery inputs, original-byte preservation, linked
  model reuse and idempotence. GPU verification uses isolated fold-3 jobs under
  `cache/fixed-refit-diagnostics/repaired_jobs/`; shortened latency repetitions
  mark those jobs as diagnostics, and they are never reused as study results.
  All four finalists reproduced every canonical metric and passed native-model
  reload checks with zero score difference on the eight checked test images.
  `cache/fixed-refit-diagnostics/repair_verification.json` records these checks
  and source hashes. Recovery preparation verified and reused all 150 neural
  jobs; the 40 complete fixed benchmarks have not been launched by the assistant.

### Completed matched study and statistical analysis

- The author executed recovery and analysis on the same Linux / RTX 4070 Ti
  environment. `results/fine_tuning_recovered/run_metadata.json` records
  `completed: true`; all 190 job manifests are completed: 150 neural fits across
  five CNNs, ten original folds and three seeds, plus 40 fixed-feature benchmarks.
  All strict fixed-refit canonical metric comparisons passed. The original run
  is retained as the audited source of the 150 reused neural jobs.
- Outputs include 150 neural fold metric rows, 5,130 neural image prediction
  rows, 40 fixed fold metric rows, 1,368 fixed image predictions, and 190 cost
  records. Statistical analysis validated predictions/fold metrics, shared
  adaptation images and the original prespecified configuration before writes.
- `manuscript_tables/analysis_manifest.json` records 14 verified input hashes,
  five neural models, four fixed procedures, 20 pairings, 10,000 paired
  class-stratified image bootstrap replicates (seed 1980), and the unchanged
  substitution margin of three percentage points. There are 16 CSV tables,
  16 LaTeX fragments and one figure exported as PNG/PDF.
- Mean fold accuracy, averaging the three single-seed metrics within each
  neural fold: EfficientNetB3 92.78%, EfficientNetB4 92.49%, MobileNet 90.55%,
  ResNet50 83.64%, ResNet101V2 71.75%. The fixed ResNet101V2/logistic finalist
  has 94.16% mean fold accuracy. These values are not seed-ensemble accuracies.
  Finalist selection used the same folds, so comparative inference remains
  exploratory. No grouped nested CV result or grouped-fold fine-tuning is claimed.
- Archived verification snapshots in `results/fine_tuning_recovered/verification/`
  distinguish the historical diagnosis and recovery preparation from the final
  completed state. They retain original report bytes and source/artifact hashes.
  Final verification checked 1,050 completed-job artifact digests (including
  local binaries), 640 canonical metric comparisons across the 40 fixed jobs,
  all 190 saved reload-check results and all 14 statistical-analysis input hashes.
  Git stores the report artifacts for both runs; `.keras`, `.h5` and `.joblib`
  model/checkpoint/pipeline binaries remain local and are excluded by scoped
  ignore rules. Their original manifest hashes are preserved. Statistical
  analysis can be reproduced from a clone; deployment or recovery additionally
  requires those matching binary artifacts.

## Release record

When this repository is released, record the release tag, commit identifier,
and archival DOI below:

- Release tag: pending
- Commit: pending
- DOI: pending


## 2026-10-06 — nested CV e conjunto externo, etapa 1

Especificações lidas na ordem: PROMPT_NESTED_CV.md e PROMPT_NESTED_CV_EXTERNAL.md; complemento prevalece. Antes de ler/implementar, a pasta externa foi adicionada ao .gitignore; as 1.929 imagens que vieram no pull foram retiradas apenas do índice (continuam locais e no histórico anterior). Inventário com hashes versionado.

A seleção original foi movida sem alterar suas funções de ranking/desempate para src/selection.py; a mesma função pura atende tabelas, nested CV e seleção final. Regressão full_search reproduz os quatro finalistas e o vencedor global B/EfficientNetB3/PCA300/linear SVM. Nenhum resultado full_search foi refeito.

Decisões explícitas registradas antes da avaliação: inviabilidade de PCA/UMAP-250/300 em treinos internos (~245 imagens), sem truncamento; adaptador sigmoide independente ajustado só ao OOF VIA para probabilidades dos fixos, mantendo SVM probability=False; bootstrap com grupos inteiros e terceiro estrato de cenas mistas; validação final FT por grupos comuns às seeds; EfficientNetB4 batch 8. As decisões estão detalhadas em EXTERNAL_TEST_PROTOCOL.md e nos novos configs.

Somente diagnósticos CPU e limpeza foram executados. Diagnósticos de pipelines/CNNs usam pequenos subconjuntos VIA, uma época e uma seed; fixtures de avaliação/curva não abrem imagens externas. Nunca são modelos finais ou evidência científica. Não houve classificação de imagem externa nem nested completa.

Limpeza: 1.929 originais → 1.576 novas (844 clear/732 nonclear); 350 cópias VIA (quatro com classe trocada), três duplicatas internas; oito novas sinalizadas por nove ligações pHash ao VIA, sem ligação semântica/geometria confirmada.População provisória principal: 1.568 (836 clear/732 nonclear). Há 1.378 grupos nas novas. Mosaicos de nove pares e 100 imagens disponíveis, relatório permanece paused_for_author_review, approved_by/date=null. Revisão pode rejeitar ligações em CSV com justificativa; o grafo automático/config são preservados. Nenhuma correção/aprovação do autor foi preenchida.

Ambiente de treino .venv e fontes artifact.py/fine_tuning.py/fixed_refit.py preservados; torch 2.7.1+cpu/transformers 4.51.3/huggingface-hub 0.30.2 instalados só no ambiente auxiliar cache/scene-groups-venv. Resultados protegidos e configs anteriores conferidos contra inventário anterior em /tmp/nested-external-protected-baseline.json; ver relatório versionado da etapa para as verificações. Protocolo e config externo fazem parte do mesmo commit de implementação. Sem execução posterior nesta entrega.

## 2026-10-06 — etapa 2: revisão do autor e desvio para teste geográfico

Autorização: `PROMPT_STAGE2_NESTED_EXTERNAL.md`, decisões do autor de 06/10/2026, continuando `fb1d7f1`. O autor rejeitou as nove ligações pHash, sem mudar o grafo automático ou os limiares. O CSV fornecido de rejeições é preservado. Revisou as 100 imagens e manteve os rótulos: correções só com cabeçalho, discordância 0/100 e Wilson 95% [0; 0,03699349820698568]. Aplicação da revisão: zero exclusões por cena, principal 1.576 (844 clear/732 nonclear), mesmos 1.378 grupos; aprovação gravada como Jefferson Antonio Ribeiro Passerini (author), 2026-10-06, com hashes atuais. Registro em `results/stage2_verification/author_review_application.json`.

Procedência declarada pelo autor: VIA, Fabricio Breve, 2020, Rio Claro/SP, 342 imagens; extensão, Jefferson A. R. Passerini, coleta posterior em Fernandópolis e Jales/SP, cerca de 400 km de Rio Claro. Definição de classe idêntica. Outra região/coletor/período; protocolo de hash/cenas e revisão visual não identificaram cenas em comum. Período exato/aparelho da extensão são `TODO(autor)` e não bloqueiam a execução. Só VIA é usado na busca/seleção e nos limiares; extensão entra no teste geográfico principal e na curva secundária.

Desvio decidido antes de observar classificações externas: abandonar nested obrigatória devido às 900–2.702 h previstas por versão, com aproximadamente 1.466 das 1.536 h seriais em preparação dominada por Relief-F. Principal passa a teste externo único dos finalistas/vencedor escolhidos na grade integral de 12.656 configurações VIA, incluindo Relief-F. O protocolo versão 2 e a configuração são commitados juntos antes do treino/avaliação. Métricas, ICs, margem 3 p.p., dois pontos de operação, publicação independentemente do resultado e curva secundária continuam em vigor. Nested é só suplementar opcional, sem autorização para execução completa.

Ausência de ajuste no teste original: `src/artifact.py` inclui fold na chave `ReliefRankingCache`, ajusta PCA/UMAP sobre x_train, Relief-F sobre x_train/y_train e classificador sobre x_train/labels[train]. O otimismo está na seleção do finalista sobre as mesmas estimativas 10-fold. OOF dos fixos agora vem dos 40 refits canônicos existentes em `results/fine_tuning_recovered`, reproduzidos usando `src/fixed_refit.py`, com folds, SHA, cobertura e métricas verificadas. Não há novos treinos para produzir esse OOF. Adaptador e limiar daí derivados herdam otimismo; o externo mede a manutenção do alvo de recall 0,95. A calibração VIA é uma referência otimista, já que o adaptador foi ajustado nessas mesmas observações OOF.

Correção complementar solicitada: média por fold de máximo da abordagem menos finalista, subtraída da média do finalista, para accuracy e recall de obstruídos; vencedor usa a grade toda. Versão conservadora de Tibshirani & Tibshirani (2009), com recall como extensão secundária, sem garantia de estimativa independente. Usar apenas métricas existentes, sem treino. Registrar diferença PCA−melhor Relief-F (melhor média de accuracy), diagnósticos por fold, CSV e LaTeX. Acurácias corrigidas de verificação: A 90,0672%, B 88,6218%, C 89,4790%, D 89,1765%.

Imagens públicas: por decisão do autor, os 1.929 JPG permanecem no histórico público do commit `4a5a1a5`. Não reescrever esse histórico. O índice atual ignora o diretório bruto; arquivos permanecem locais. Modelos, pipelines e checkpoints são locais/ignorados, acompanhados por manifests e hashes versionados.

Suporte operacional antes da avaliação: reamostragem por cena precomputada conserva os mesmos sorteios estratificados e todas as multiplicidades de membros, sem reconstruir grupos em cada réplica. ICs para recall/FNR da média das seeds e ensemble usam cenas inteiras; Wilson fica por seed individual. Congelamento completo é idempotente: repetir o runner verifica hashes e reutiliza artefatos, sem recalcular benchmarks nem trocar o hash dos modelos já congelados. Ambiente CUDA é preparado só nas variáveis dos processos filhos; nenhuma instalação ou mudança no ambiente de treino.

### 2026-10-07 — recuperação operacional de memória, sem mudança científica

O ajuste final EfficientNetB4/seed 1980 já havia encerrado suas 16 épocas quando a verificação nativa de recarga em oito imagens esgotou a GPU. Reiniciar o processo reutilizou o checkpoint parado, sem novas épocas; batch de treinamento 8 e decisões de recarga idênticas. Os 19 modelos foram congelados e a avaliação externa única terminou com 29.944 linhas, hash SHA-256 `53231b8355530ce518ff86581f6cbd6a1baaec25dcaf2b2cc4e2275f83b0bfe1`.

A curva secundária encontrou o mesmo erro após 19 ajustes concluídos. O runner operacional passa a permitir no máximo cinco retomadas em processo novo, exclusivamente quando o traceback identifica a verificação de recarga e o checkpoint confirma early stopping já concluído. Registra o manifesto anterior, hash do checkpoint, batch efetivo antes das reduções espúrias de recarga e hashes dos resultados principais em `reload_recovery.json` e `progress.json`. Nenhuma fonte científica, configuração, limiar, checkpoint de pesos ou predição principal é alterada; não há ajuste por resultado externo. Erros fora dessa condição interrompem o fluxo e a publicação. A auditoria final exige todos os resultados, pytest completo e a preservação dos arquivos e pacotes protegidos.

### Conclusão da execução geográfica — 2026-10-07T18:52:22.340905+00:00

Pipeline completo no tmux: 4 fixos + 15 FT, avaliação única em 1.576 externas (29.944 linhas sem duplicatas), análise por cenas e curva secundária concluídas. Commit do protocolo/desvio: `cb7c510e2f649ead6284bb6b033fe8356f3f22b1`. Fonte científica, 4.103 arquivos protegidos e 69 versões de pacotes preservados. Relatório completo e testes finais em docs/STAGE2_REPORT.md e results/stage2_verification/. Modelos binários locais; manifests/hashes/resultados versionados. Nenhuma nested completa executada.

## Análises pós-teste — 08/10/2026

Etapa 3, continuando b64221f: todas as análises são `exploratory/post hoc`, posteriores ao teste externo; não substituem resultados principais nem alteram modelos, limiares ou predições congeladas. Ordem autorizada: 0 → 4 → 3 → 2 → 1 → 5. Novas saídas exclusivamente em results/stage3/ e pacote cego em external-data/blind_relabel/. Nomes de clusters e categorias de erro ficam vazios; nenhum rótulo será corrigido e nenhuma resposta humana será inventada.

Limiar de baixa luz, definido sem consultar erros: percentil 10 das 342 médias Y (YCbCr) do VIA = **93.746906000000**, escala 0–255. Baixa luz: média Y < limiar. Saturação HSV /255; contraste RMS = desvio-padrão de Y; Canny 100/200, apertureSize=3, L2gradient=True; nitidez = variância do Laplaciano CV_64F, ksize=1. Distâncias cosseno reutilizam DINOv2 e features ImageNet congeladas, sem nova inferência; para VIA, o próprio vizinho é excluído. Quintis definidos nas externas; k-means 12, seed 1980, n_init=20, embeddings normalizados.

ICs: 10.000 sorteios de cenas inteiras nos estratos nonclear/clear/mistos, seed 1980; FT principal = média das métricas individuais das seeds, ensemble apenas sensibilidade. Oráculo externo não implantável: limiar de maior acurácia, desempate pelo menor limiar, livre iff score_clear >= limiar. Especificidade a recall fixo usa limites externos e é reestimada em cada bootstrap. ROC obstruído positivo usa a ordem exata de -score_clear, matematicamente equivalente a 1-score_clear, evitando cancelamento em probabilidades próximas de zero. AUC parcial em sensibilidade [0,90;1,00] integra especificidade nessa faixa, referência aleatória 0,005, máximo 0,10, correção McClish: 0,5×[1+(área−0,005)/0,095]. [Documentação e implementação pROC](https://rdrr.io/cran/pROC/man/auc.html). Contrastes de AUC fixo−média das seeds usam bootstrap pareado por cenas e Holm; [DeLong](https://pubmed.ncbi.nlm.nih.gov/3203132/) multivariado é sensibilidade, com contraste [1,−1/3,−1/3,−1/3] e sem ajustar a correlação entre cenas. Regressões logísticas do acerto são descritivas, por seed, máxima verossimilhança sem penalização e covariância sandwich por cena, correção finita G/(G−1)×(N−1)/(N−p). Platt FT: apenas OOF VIA por seed, logit com clip 1e−7, C=1, max_iter=500, seed 1980; não é ajuste no externo. Não há nova seleção de configuração. FT agrupado será só estimado, sem execução.
