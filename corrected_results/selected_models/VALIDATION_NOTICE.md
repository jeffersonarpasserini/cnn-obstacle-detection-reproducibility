# Validation notice

The archived rows for Approaches A, B, and C in this directory passed the
structural and numerical validation performed after the Linux execution.

The archived Approach D row named
`approach_d_mobilenet_resnet50_pca100_rbf_svm` is obsolete. It was produced
with 100 total components (50 per CNN), whereas the methodology retains 100
components from each CNN and concatenates them into a 200-component vector.

Use `configs/approach_d_corrected.json` and write the replacement result to
`corrected_results/approach_d_corrected/`. Do not cite the obsolete Approach D
row in the manuscript.
