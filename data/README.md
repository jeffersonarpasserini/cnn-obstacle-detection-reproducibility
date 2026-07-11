# VIA dataset

The artifact includes a local copy of the 342 images in `via-dataset/`. The canonical source remains the dataset authors' repository.

Canonical source: <https://github.com/fbreve/via-dataset>

Expected layout:

```text
via-dataset/clear.000.jpg
via-dataset/nonclear.000.jpg
...
```

The corrected runner sorts filenames before assigning labels and folds. Files beginning with `clear.` are assigned to the clear-path class; files beginning with `nonclear.` or `non-clear.` are assigned to the obstructed-path class. Any other filename is rejected to prevent silent label errors.

When using the dataset, cite the original IJCNN 2020 publication identified in the dataset repository.

Run `python scripts/verify_dataset.py` to validate the number of images, class distribution, filenames, and SHA-256 hashes against `data/dataset_manifest.csv`.
