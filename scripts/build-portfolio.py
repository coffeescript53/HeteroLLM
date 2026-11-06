#!/usr/bin/env python3
"""Build the static portfolio and source download locally; no external services."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "build", "dist", "__pycache__", ".venv", "node_modules",
            "private", "runs", "models", "weights", "llama.cpp", "exports",
            "experiments", "audits", "studio"}
EVIDENCE = ("results.json", "recorder-summary.json", "raw-timings.json")


def no_symlinks(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Symlink is not allowed: {path}")


def source_file(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe source path: {name}")
    path = ROOT.joinpath(*relative.parts)
    no_symlinks(path)
    if not path.resolve().is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Source must be a regular file inside this repository: {name}")
    return path


def source_names() -> list[str]:
    if (ROOT / ".git").exists():
        no_symlinks(ROOT / ".git")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        command = ["git", "-C", str(ROOT)]
        top = subprocess.check_output(command + ["rev-parse", "--show-toplevel"], env=env)
        if Path(os.fsdecode(top).strip()).resolve() != ROOT:
            raise ValueError("Refusing to use a parent repository")
        raw = subprocess.check_output(command + ["ls-files", "-z"], env=env)
        return [os.fsdecode(name) for name in raw.split(b"\0") if name]
    manifest = json.loads(source_file("PUBLIC_SOURCE_MANIFEST.json").read_text())
    return [entry["path"] for entry in manifest["files"]] + ["PUBLIC_SOURCE_MANIFEST.json"]


def destination(output: Path, name: str) -> Path:
    path = output / name
    no_symlinks(path)
    if path.exists() and not path.is_file():
        raise ValueError(f"Destination is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    output = parser.parse_args().output
    if not output.is_absolute():
        parser.error("--output must be an absolute directory path")
    no_symlinks(output)
    output = output.resolve()
    if ROOT.is_relative_to(output):
        parser.error("Output cannot be the source root or one of its ancestors")
    selected = []
    for name in sorted(set(source_names())):
        parts = PurePosixPath(name).parts
        if any(part in EXCLUDED or part.startswith("runtime-c2-") for part in parts):
            continue
        path = source_file(name)
        if (path.name == ".env" or path.name.startswith(".env.")
                or set(path.suffixes) & {".gguf", ".safetensors", ".bin", ".so", ".pyc", ".pyo"}):
            continue
        if path.is_relative_to(output):
            raise ValueError(f"Output would overlap selected source: {name}")
        selected.append((name, path))
    if not selected:
        raise ValueError("No tracked or manifest-listed source files were found")
    copies = {"index.html": source_file("site/index.html"),
              **{f"evidence/{name}": source_file(f"evidence/portfolio/{name}") for name in EVIDENCE}}
    for name, source in copies.items():
        shutil.copy2(source, destination(output, name))
    archive = destination(output, "downloads/HeteroLLM-Source.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, source in selected:
            info = zipfile.ZipInfo("HeteroLLM/" + name, (2026, 11, 4, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(source.stat().st_mode)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, source.read_bytes())
    print(f"Built {output} with {len(selected)} source files in {archive.name}")


if __name__ == "__main__":
    main()
