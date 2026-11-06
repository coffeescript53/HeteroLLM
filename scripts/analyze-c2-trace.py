#!/usr/bin/env python3
"""Offline CPU trace inspection only; never loads a model or launches a runtime."""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heterollm.trace_analysis import analyze, strict_json, integer, union_ns

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--stderr', required=True, type=Path)
    a = p.parse_args()
    if a.trace.stat().st_size > 64 * 1024**2:
        p.error('trace exceeds offline 64 MiB cap')
    print(json.dumps(analyze(a.trace.read_text(), a.stderr.read_text()), indent=2))


if __name__ == '__main__':
    main()
