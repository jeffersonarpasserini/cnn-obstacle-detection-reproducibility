#!/usr/bin/env python3
"""Validate archived result counts and headline medians."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")


def first(pattern: str) -> Path:
    matches = sorted(glob.glob(str(RESULTS / pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[0])


def model_key(frame: pd.DataFrame) -> str:
    for candidate in ("Metodo", "metodo", "MetodoABR"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("No model identifier column found")


def main():
    expected = {"A": 272, "B": 7200, "C": 2592, "D": 2592}
    patterns = {
        "A": "Abordagem01/Dados/*data_detailed*full*.csv",
        "B": "Abordagem02/Dados/*data_detailed*.csv",
        "C": "Abordagem03/Dados/*data_detailed*.csv",
        "D": "Abordagem04/Dados/*data_detailed*.csv",
    }
    for approach, pattern in patterns.items():
        frame = load(first(pattern))
        key = model_key(frame)
        configurations = frame[key].nunique()
        folds = frame["Fold"].nunique()
        duplicates = int(frame.duplicated([key, "Fold"]).sum())
        print(
            f"Approach {approach}: configurations={configurations}, "
            f"folds={folds}, duplicate model-fold rows={duplicates}"
        )
        assert configurations == expected[approach]
        assert folds == 10
        assert duplicates == 0
    print("Legacy result structure: OK")


if __name__ == "__main__":
    main()

