#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODES = ["gray", "archive", "color"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scanario on all test images and save ordered outputs.")
    parser.add_argument("--tests-dir", default="tests", help="Directory containing input test images")
    parser.add_argument("--results-dir", default="results", help="Directory where outputs are written")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES, help="Enhancement modes to run")
    parser.add_argument("--backend", choices=["auto", "nano", "rembg"], default="auto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tests_dir = (root / args.tests_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in tests_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if not images:
        print(f"No test images found in {tests_dir}", file=sys.stderr)
        return 1

    for mode in args.modes:
        mode_dir = results_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== mode={mode} backend={args.backend} ===")
        for image in images:
            print(f"\n--- {image.name} ---")
            cmd = [
                sys.executable,
                str(root / "scanario.py"),
                "scan",
                str(image),
                "--out-dir",
                str(mode_dir),
                "--mode",
                mode,
                "--backend",
                args.backend,
            ]
            if args.debug:
                cmd.append("--debug")
            subprocess.run(cmd, check=True)

    print(f"\nDone. Results in {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
