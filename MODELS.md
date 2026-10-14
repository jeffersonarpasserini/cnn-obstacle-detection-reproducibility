# Trained models for v2.0.0

The exact frozen binaries used for the external evaluation are distributed as **GitHub release attachments**, outside Git. They are **not included in the Zenodo source-code archive** of this release. Keep these attachments with the code when preserving a complete executable artifact. At this review stage the packages are prepared locally; the URLs below become available when v2.0.0 is published.

All 28 files were authenticated against `results/external_test/final_models.json`: 15 fine-tuned models, four fixed pipelines, four encoders and five fixed-model probability-adapter files. `adapter_winner.joblib` is the B-winner alias and has the same hash as `adapter_B.joblib`. Fine-tuned models emit native probabilities; no additional FT binary adapter appears in the final frozen manifest. Exploratory post hoc calibration adapters, intermediate checkpoints, last/best training weights, diagnostic/smoke models and caches are excluded.

## Packages

| Package | Contents | Bytes (MiB) | SHA-256 |
| --- | --- | ---: | --- |
| [models_encoders_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_encoders_v2.0.0.tar.gz) | 4 frozen ImageNet encoders: EfficientNetB3, MobileNet, ResNet50, ResNet101V2 | 298260171 (284.44) | `ae7c60a3cb6849c32dbaf7f7c696035e177f2594e2f4b5cdedc525dede2f78aa` |
| [models_fixed_pipelines_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_fixed_pipelines_v2.0.0.tar.gz) | 4 fixed pipelines A–D; 5 sigmoid adapter files, including winner alias | 510730052 (487.07) | `ab75f6b23fa5123eeeb37caa35b5889e3966050284793ebc29a2de42c9172135` |
| [models_ft_EfficientNetB3_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_ft_EfficientNetB3_v2.0.0.tar.gz) | Final EfficientNetB3 models, seeds 1980/1981/1982; native probabilities (no separate frozen FT adapter) | 122546136 (116.87) | `c8e928a8ca1a4ce66973eb7e355b2f2bbb409e6ccd02087fa7db5d968d45d9e0` |
| [models_ft_EfficientNetB4_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_ft_EfficientNetB4_v2.0.0.tar.gz) | Final EfficientNetB4 models, seeds 1980/1981/1982; native probabilities (no separate frozen FT adapter) | 199766566 (190.51) | `d91a3f97e12ff4d968ebe554817a52575236dfd451405cf5cbde18ec89b7de44` |
| [models_ft_MobileNet_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_ft_MobileNet_v2.0.0.tar.gz) | Final MobileNet models, seeds 1980/1981/1982; native probabilities (no separate frozen FT adapter) | 37539438 (35.80) | `87401e0b0b5b62fa5d0dadbc459999a7868d4a2293f5ec0474d15fd6449fe9b6` |
| [models_ft_ResNet101V2_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_ft_ResNet101V2_v2.0.0.tar.gz) | Final ResNet101V2 models, seeds 1980/1981/1982; native probabilities (no separate frozen FT adapter) | 478156367 (456.01) | `00db33de4f3211e39a08910dc978ca0cda2db527d1c086c1f47cd2f2bae3770a` |
| [models_ft_ResNet50_v2.0.0.tar.gz](https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/models_ft_ResNet50_v2.0.0.tar.gz) | Final ResNet50 models, seeds 1980/1981/1982; native probabilities (no separate frozen FT adapter) | 265443709 (253.15) | `7f1b97873589af09ac14a92f0522e4044e81cd782f1a54817a861354f5404179` |

Each download link has the form `https://github.com/jeffersonarpasserini/cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/<package>`. The release also provides SHA256SUMS.txt. `models_packages.json` records archive sizes/hashes/URLs; `models_manifest.csv` records every repository-relative member, its byte size, SHA-256, model ID, seed and role.

## Verify and extract

Run from the repository root with Python 3.12 or later; fetching requires only the standard library. Download all seven packages and SHA256SUMS.txt into a sibling assets directory, then use:

```bash
cd ../assets_v2.0.0
sha256sum -c SHA256SUMS.txt
cd ../cnn-obstacle-detection-reproducibility
python scripts/fetch_models.py --from-dir ../assets_v2.0.0 --verify-only
python scripts/fetch_models.py --from-dir ../assets_v2.0.0
python scripts/fetch_models.py --verify-only
```

Alternatively, `python scripts/fetch_models.py` downloads the packages from the version-specific URLs. The script authenticates each complete archive and every uncompressed file against the versioned package metadata, models_manifest.csv and final_models.json before installation. It rejects missing/extra/duplicate members, links, unsafe paths and different existing model files. `--verify-only` alone checks installed binaries offline; with `--from-dir` it checks the archives without installing. No model is deserialized, trained or evaluated by this script.

The archives contain only regular files with paths relative to the repository root; extraction restores `results/external_test/models/...`, `encoders/...` and `adapter_*.joblib` at their frozen locations. Use the repository dependency snapshots for subsequent scientific execution.

## Correspondence to article results

| Binaries/model IDs | Frozen evidence | Revised article |
| --- | --- | --- |
| Fixed A: ResNet101V2/logistic; B: EfficientNetB3/PCA300/linear SVM; C: joint MobileNet+ResNet50/PCA300; D: per-CNN PCA300/MobileNet+ResNet50 | `final_models.json`, `thresholds.json`, `fixed_oof_predictions.csv`, `models/<model_id>/identity.json` and `complete.json` under `results/external_test/` | Tables 3–8 select and compare the fixed procedures; Table 10 evaluates their frozen external deployment |
| ft_EfficientNetB3, ft_EfficientNetB4, ft_MobileNet, ft_ResNet50, ft_ResNet101V2, each with seeds 1980–1982 | `results/fine_tuning_recovered/manuscript_tables/`; `results/external_test/models/ft_<architecture>__seed_<seed>/`; `results/external_test/predictions.csv` | Table 9 documents matched VIA fine-tuning; Table 10 uses the VIA-only final refits distributed here; Table 12 reports measured costs |
| Four shared encoders and five fixed adapters | `final_models.json` (`encoders` and `artifacts`), `thresholds.json`, `calibrated_via_oof.csv` | Table 10; encoders/adapters were frozen using VIA only |

These are the final VIA-only external-evaluation binaries, not all outer-fold training checkpoints behind Table 9. Historical manuscript scores used original evaluated external images before privacy anonymization; the distributed images and frozen historical predictions are documented separately.

## ImageNet base-weight licenses and provenance

Official sources were checked for each family. The weight-origin terms below are separate from this repository's MIT software and CC BY 4.0 external-data licenses. The packaged fine-tuned weights inherit their base-weight attribution; do not relabel all model assets as CC BY 4.0.

| Base family | Upstream weight-origin terms | Official evidence |
| --- | --- | --- |
| ResNet50 | MIT | [Keras historical weight-license documentation](https://github.com/keras-team/keras/blob/2.3.1/docs/templates/applications.md#resnet) identifies Kaiming He's weights and their [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE). |
| ResNet101V2 | BSD 3-Clause | [Keras ResNetV2 weight-license documentation](https://github.com/keras-team/keras/blob/2.3.1/docs/templates/applications.md#resnet) identifies Facebook's source; [official Facebook BSD license](https://github.com/facebookarchive/fb.resnet.torch/blob/master/LICENSE). |
| MobileNet (v1) | Apache-2.0 | [Keras MobileNet weight-license documentation](https://github.com/keras-team/keras/blob/2.3.1/docs/templates/applications.md#mobilenet), [TensorFlow model/checkpoint source](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), and [official license](https://github.com/tensorflow/models/blob/master/LICENSE). |
| EfficientNetB3 and EfficientNetB4 | Apache-2.0 upstream distribution | [Official Keras loader and weight hashes](https://github.com/keras-team/keras/blob/v2.6.0/keras/applications/efficientnet.py), [TensorFlow official EfficientNet checkpoints](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), and [upstream Apache-2.0 license](https://github.com/tensorflow/tpu/blob/master/LICENSE). |

For EfficientNet, the cited evidence establishes the upstream implementation/checkpoint distribution and its repository license; the Keras loader does not contain a separate weight-specific license grant. This qualification is retained rather than inferred from the artifact's MIT LICENSE. Current [Keras](https://github.com/keras-team/keras/blob/master/LICENSE) itself is Apache-2.0; its historical applications package license alone does not determine every base-weight license. Preserve the cited upstream notices when redistributing the binaries.

Deterministic packaging uses sorted explicit member names, GNU tar format, UTC mtime 2026-10-14 00:00:00, uid/gid 0, mode 0644 and gzip without filename/timestamp metadata. Every compressed stream was built twice and matched byte-for-byte. No training, inference or modification of a frozen binary was needed.
