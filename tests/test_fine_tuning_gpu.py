"""Two independent five-architecture Linux/GPU smokes, never CPU fallback."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.fine_tuning import compare_smoke_runs, completed_job
from scripts.configure_tensorflow_gpu import gpu_environment


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu


def run_stage(arguments, log_path):
    command = [sys.executable, "src/fine_tuning.py", "--config", "configs/fine_tuning.json",
               *arguments]
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=gpu_environment())
    if result.returncode:
        pytest.fail(
            f"GPU stage failed (exit {result.returncode}). Log: {log_path}\n"
            + log_path.read_text()[-6000:]
        )


def verify_gpu_smoke(directory, architectures):
    report = json.loads((directory / "smoke_report.json").read_text())
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert report["completed"] is True and metadata["completed"] is True
    assert report["smoke_all"] is True and metadata["smoke"] is True
    assert report["runtime"]["platform"].startswith("Linux")
    assert report["runtime"]["training_device"] == "GPU:0"
    assert report["runtime"]["benchmark_devices"] == ["cpu", "gpu"]
    assert report["folds"] == [1] and report["seeds"] == [1980]
    assert report["requested_architectures"] == architectures
    assert set(report["architectures"]) == set(architectures)
    assert report["config"]["epochs"] == 1
    assert report["config"]["fixed_features"]["enabled"] is False
    costs = pd.read_csv(directory / "computational_costs.csv")
    assert len(costs) == len(architectures)
    assert costs["effective_epochs"].eq(1).all()
    assert costs["family"].eq("fine_tuning").all()
    for architecture, row in report["architectures"].items():
        assert row["status"] == "passed", architecture
        assert row["epoch_seconds"] > 0 and row["batch_size"] > 0, architecture
        assert row["peak_gpu_memory_status"] == "measured", architecture
        assert row["peak_gpu_memory_bytes"] > 0, architecture
        assert row["peak_host_memory_bytes"] > 0, architecture
        assert row["determinism"]["enabled"] is True, architecture
        assert row["reload_check"]["status"] == "passed", architecture
        assert row["reload_check"]["image_count"] == 8, architecture
        assert row["reload_check"]["max_abs_difference"] <= 1e-5, architecture
        assert row["reload_check"]["decisions_identical"] is True, architecture
        for device in ("cpu", "gpu"):
            assert row["latency"][f"{device}_latency_status"] == "measured", architecture
            assert row["latency"][f"{device}_latency_median_ms"] > 0, architecture
        assert row["latency"]["latency_batch_size"] == 1, architecture
        directory_job = directory / "jobs" / f"ft_d_{architecture.lower()}__fold_01__seed_1980"
        manifest = json.loads((directory_job / "manifest.json").read_text())
        assert completed_job(directory_job, manifest["request"]), architecture
        assert "model.keras" in manifest["artifacts"], architecture
        checkpoint = directory_job / manifest["checkpoint"]
        assert (checkpoint / "last.keras").is_file(), architecture
        assert (checkpoint / "best.weights.h5").is_file(), architecture
    return report


def test_five_architecture_gpu_smoke_and_independent_repeat(request, tmp_path):
    assert sys.platform == "linux", "GPU certification requires Linux"
    configured = request.config.getoption("--gpu-output-dir")
    output = (ROOT / configured).resolve() if configured else tmp_path / "gpu_smokes"
    assert not output.exists(), "Use a fresh --gpu-output-dir for independent GPU runs"
    output.mkdir(parents=True)
    config = json.loads((ROOT / "configs/fine_tuning.json").read_text())
    architectures = config["architectures"]
    run_stage(["--runtime-info", "--device", "gpu"], output / "runtime.log")
    run_stage(["--prefetch-weights", "--output-dir", str(output / "prefetch")],
              output / "prefetch.log")
    first, repeat = output / "smoke_all", output / "smoke_all_repeat"
    arguments = ["--dataset", "via-dataset", "--device", "gpu", "--smoke-all"]
    run_stage([*arguments, "--output-dir", str(first)], output / "smoke_all.log")
    verify_gpu_smoke(first, architectures)
    run_stage([*arguments, "--output-dir", str(repeat), "--compare-smoke", str(first)],
              output / "smoke_all_repeat.log")
    report = verify_gpu_smoke(repeat, architectures)
    assert report["repeat_check"]["status"] == "passed"
    assert report["repeat_check"]["identical_predictions"] is True
    assert all(row["determinism"]["status"] == "passed_exact_independent_repeat"
               for row in report["architectures"].values())
    assert compare_smoke_runs(repeat, first)["identical_predictions"] is True
    print(f"Linux/GPU smoke reports and logs: {output}")
