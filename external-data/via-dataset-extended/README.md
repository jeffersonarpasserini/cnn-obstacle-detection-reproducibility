# External path-obstruction test set

This directory contains exactly 1,576 released JPEG images: 844 clear paths and
732 obstructed paths, organized into 1,378 scene groups. The labels and scene
groups are in `../external_manifest_clean.csv` and `../external_groups.csv`.
Images have their original 750 × 1000 pixel dimensions.

## Collection and labels

Jefferson Antonio Ribeiro Passerini collected the smartphone photographs in
Fernandópolis and Jales, São Paulo, Brazil, during 2024–2025. The extension is
geographically and temporally separate from the original VIA collection in
Rio Claro. A clear path has no obstacle in the pedestrian's path ahead;
an obstructed path has an obstacle in that path. In the manifests, clear is
label 1 and obstructed is label 0. An object outside the walking path does
not by itself make the path obstructed.

The 1,929 raw files were cleaned to 1,576 retained files. The 353 exclusions
include 350 exact matches to VIA and three internal duplicates. Four VIA
matches had conflicting original labels; they are recorded in the historical
exclusion audit and remain excluded. The author reviewed candidate scene
overlaps and the label audit. The retained labels and 1,378 scene groups are
frozen; related views stay in the same group.

## Privacy screening and anonymization

All retained images underwent local automatic face and license-plate screening,
followed by manual review by the author, including a manually added plate box.
Only approved regions received local Gaussian blur: **20 regions (19 license
plates and one face) in 16 images**. The boxes were expanded by 15% on each
side and clipped to the image bounds. The Gaussian kernel was odd and at least
51 pixels, or approximately one third of the larger expanded-box dimension.
The remaining **1,560 images are byte-identical** to their evaluated originals.
JPEG quantization tables, chroma subsampling, progressive mode, dimensions and
original metadata segments were retained. JPEG re-encoding is lossy; decoded
pixels outside the blur regions can therefore differ slightly in changed files.

The four fixed pipelines and the five matched fine-tuning architectures
(three initialization seeds each) were run on the released images with the
same frozen preprocessing, probability adapters and default/safety thresholds.
The **1,560 intact images produced exactly identical scores** for every model
and seed. The largest absolute accuracy variation per model and seed was
**0.13 percentage points** (0.126904 before rounding).

**The results published in the article were obtained using the original
evaluated images, before anonymization.** Historical predictions and result
tables retain those values. The released image hashes describe the privacy
copies; they do not replace the evaluated-image hashes.

`../external_release_manifest.csv` records filename, frozen label and group,
dimensions, SHA-256 of both evaluated and released files, anonymization flag,
and the number of blurred faces and plates. The release manifest SHA-256 is
`02eb271bd17d9a63d8d4edf7af11bd782b3cc4f165c2516175e39478c94cdd48`.
No excluded raw image or blind-relabel image copy is distributed here.

## Verification

From the repository root:

```bash
python scripts/verify_external_release.py
```

The verifier authenticates the manifest, checks every released image hash,
the class/group/anonymization counts, frozen labels and group identities,
and the absence of all 353 excluded filenames.

## License and citation

The released external images, labels, groups and annotations are licensed
under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
See `../../LICENSE-DATA` and
<https://creativecommons.org/licenses/by/4.0/legalcode>.

Credit Jefferson Antonio Ribeiro Passerini for the image collection and cite
the study by Jefferson Antonio Ribeiro Passerini and Fabricio Aparecido Breve,
“Cross-Analysis of Fixed CNN Feature Extractors for Path-Obstruction
Classification to Assist Visually Impaired People.” Citation metadata is in
`../../CITATION.cff`; the artifact concept DOI is
<https://doi.org/10.5281/zenodo.21830689>. Identify any further modifications
when redistributing the data.
