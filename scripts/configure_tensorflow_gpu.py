#!/usr/bin/env python3
"""Expose pip NVIDIA libraries/ptxas inside the active project virtualenv.

Matches https://www.tensorflow.org/install/pip#gpu_only_virtual_environment_configuration
without overwriting existing files or modifying the system CUDA installation.
"""
import importlib.util
import os
from pathlib import Path
import shlex
import sys
import sysconfig


def nvidia_paths():
    roots = {Path(sysconfig.get_path(key)).resolve() / "nvidia" for key in ("purelib", "platlib")}
    libraries = sorted({p for root in roots for p in root.glob("*/lib") if p.is_dir()})
    binaries = sorted({p.parent for root in roots for p in root.glob("*/bin/ptxas")})
    return libraries, binaries


def gpu_environment():
    """Set loader/compiler paths before child processes import TensorFlow."""
    environment = dict(os.environ)
    libraries, binaries = nvidia_paths()
    for variable, paths in (("LD_LIBRARY_PATH", libraries), ("PATH", binaries)):
        if paths:
            entries = [str(path) for path in paths]
            if environment.get(variable):
                entries.append(environment[variable])
            environment[variable] = os.pathsep.join(entries)
    return environment


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Activate the project virtualenv before configuring TensorFlow GPU")
    spec = importlib.util.find_spec("tensorflow")
    if spec is None:
        raise SystemExit("Install requirements-experiments.txt before configuring TensorFlow GPU")
    tensorflow_dir = Path(spec.origin).resolve().parent
    environment = Path(sys.prefix).resolve()
    if not tensorflow_dir.is_relative_to(environment):
        raise SystemExit("TensorFlow must be installed inside the active virtualenv")
    roots = {Path(sysconfig.get_path(key)).resolve() / "nvidia" for key in ("purelib", "platlib")}
    libraries = sorted({p for root in roots for p in root.glob("*/lib/*.so*")})
    compilers = sorted({p for root in roots for p in root.glob("*/bin/ptxas")})
    if not libraries or not compilers:
        raise SystemExit("Missing pip NVIDIA libraries/ptxas; install tensorflow[and-cuda]")
    links = [(source, tensorflow_dir / source.name) for source in libraries]
    links.append((compilers[0], environment / "bin/ptxas"))
    # Check every destination before writing any links.
    for source, target in links:
        if target.is_symlink() or target.exists():
            if target.resolve() != source.resolve():
                raise SystemExit(f"Refusing to replace an existing file/link: {target}")
    created = 0
    for source, target in links:
        if not target.is_symlink():
            target.symlink_to(os.path.relpath(source, target.parent))
            created += 1
    libraries, binaries = nvidia_paths()
    activation = environment / "bin/activate-tensorflow-gpu"
    activation.write_text(
        "# Source after activating this virtualenv.\n"
        + "export LD_LIBRARY_PATH=" + shlex.quote(os.pathsep.join(map(str, libraries)))
        + '${LD_LIBRARY_PATH:+:"$LD_LIBRARY_PATH"}\n'
        + "export PATH=" + shlex.quote(os.pathsep.join(map(str, binaries))) + ':"$PATH"\n'
    )
    print(f"TensorFlow GPU environment configured: {created} links created in {environment}")
    print(f"Activate CUDA library/compiler paths with: source {shlex.quote(str(activation))}")


if __name__ == "__main__":
    main()
