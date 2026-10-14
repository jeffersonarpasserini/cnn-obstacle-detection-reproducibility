"""Opt-in hardware integration checks, with persistent GPU smoke artifacts."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", default=False,
        help="Run real GPU integration tests; missing GPU/dependencies are failures.",
    )
    parser.addoption(
        "--gpu-output-dir", default=None,
        help="Fresh output directory for GPU smoke reports, models and logs.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires real Linux/TensorFlow GPU execution")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(reason="Use --run-gpu to execute real GPU integration tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
