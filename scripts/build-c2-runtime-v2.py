#!/usr/bin/env python3
"""Reuse the accepted bounded compilation supervisor for the new isolated tree."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('monitored_build', ROOT/'scripts/build-c2-runtime.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
build.OUT = ROOT/'runs/llama33-70b/c2-recorder-v2-prep-20261026'
build.SOURCE = ROOT/'runtime-c2-trace-v2'
build.BUILD = build.SOURCE/'build-mixed'

if __name__ == '__main__':
    raise SystemExit(build.main())
