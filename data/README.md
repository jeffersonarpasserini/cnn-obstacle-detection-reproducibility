# VIA dataset

The images are maintained by the dataset authors and are not duplicated in this artifact.

1. Download or clone: <https://github.com/fbreve/via-dataset>
2. Place the repository at `data/via-dataset/`.
3. Confirm that the images are available at `data/via-dataset/images/`.

The corrected runner sorts filenames before assigning labels and folds. Files beginning with `clear.` are assigned to the clear-path class; files beginning with `nonclear.` or `non-clear.` are assigned to the obstructed-path class. Any other filename is rejected to prevent silent label errors.

When using the dataset, cite the original IJCNN 2020 publication identified in the dataset repository.

