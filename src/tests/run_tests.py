#!/usr/bin/env python3
"""Run the scanario CLI on every image in src/tests/ and write outputs to results/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODES = ["gray", "archive", "color"]


def main() -> int:
    root = Path(__file__).resolve().parents[2]  # repo root
    default_tests = root / "src" / "tests"
    default_results = root / "results"

    parser = argparse.ArgumentParser(description="Run scanario on all test images and save ordered outputs.")
    parser.add_argument("--tests-dir", default=str(default_tests), help="Directory containing input test images")
    parser.add_argument("--results-dir", default=str(default_results), help="Directory where outputs are written")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES, help="Enhancement modes to run")
    parser.add_argument("--backend", choices=["auto", "nano", "rembg"], default="auto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in tests_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"No test images found in {tests_dir}", file=sys.stderr)
        return 1

    env_python = sys.executable
    src_dir = root / "src"

    for mode in args.modes:
        mode_dir = results_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== mode={mode} backend={args.backend} ===")
        for image in images:
            print(f"\n--- {image.name} ---")
            cmd = [
                env_python,
                "-m",
                "scanario.main",
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
            env = {"PYTHONPATH": str(src_dir), **__import__("os").environ}
            subprocess.run(cmd, check=True, env=env)

    print(f"\nDone. Results in {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
