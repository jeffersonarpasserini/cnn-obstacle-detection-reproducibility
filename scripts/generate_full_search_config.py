#!/usr/bin/env python3
"""Generate the 12,656-configuration search-space JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CNN = [
    "Xception", "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152",
    "ResNet50V2", "ResNet101V2", "ResNet152V2", "InceptionV3",
    "InceptionResNetV2", "MobileNet", "DenseNet121", "DenseNet169",
    "DenseNet201", "NASNetMobile", "MobileNetV2", "EfficientNetB0",
    "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
    "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
]
PAIRS = [
    ["EfficientNetB1", "EfficientNetB5"],
    ["MobileNet", "ResNet101"],
    ["ResNet101", "DenseNet169"],
    ["ResNet101", "DenseNet121"],
    ["ResNet101", "MobileNetV2"],
    ["EfficientNetB0", "MobileNet"],
    ["MobileNet", "ResNet50"],
    ["Xception", "ResNet50"],
    ["VGG16", "VGG19"],
]
REDUCTIONS = ["pca", "umap", "relieff"]
COMPONENTS = [2, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300]
CLASSIFIERS = [
    "decision_tree", "rbf_svm", "linear_svm", "mlp", "logistic",
    "random_forest", "adaboost", "gaussian_nb",
]


def slug(parts):
    return "_".join(str(part).lower() for part in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/full_search.json")
    args = parser.parse_args()
    experiments = []

    for extractors in [[name] for name in CNN] + PAIRS:
        for classifier in CLASSIFIERS:
            experiments.append({
                "name": slug(["a", *extractors, classifier]),
                "approach": "A", "extractors": extractors,
                "reduction": "full", "components": None,
                "classifier": classifier, "scale": False,
            })
    for extractor in CNN:
        for reduction in REDUCTIONS:
            for components in COMPONENTS:
                for classifier in CLASSIFIERS:
                    experiments.append({
                        "name": slug(["b", extractor, reduction, components, classifier]),
                        "approach": "B", "extractors": [extractor],
                        "reduction": reduction, "components": components,
                        "classifier": classifier, "scale": False,
                    })
    for approach in ("C", "D"):
        for extractors in PAIRS:
            for reduction in REDUCTIONS:
                for components in COMPONENTS:
                    for classifier in CLASSIFIERS:
                        experiments.append({
                            "name": slug([approach, *extractors, reduction, components, classifier]),
                            "approach": approach, "extractors": extractors,
                            "reduction": reduction, "components": components,
                            "classifier": classifier, "scale": False,
                        })

    assert len(experiments) == 12656, len(experiments)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({
            "seed": 1980,
            "n_splits": 10,
            "protocol": "stratified_kfold",
            "experiments": experiments,
        }, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(experiments)} configurations to {output}")


if __name__ == "__main__":
    main()

