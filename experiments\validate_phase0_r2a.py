from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase0_r2a")
    args = parser.parse_args()
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = (
        f"command: {sys.executable} -m pytest -q\n"
        f"returncode: {process.returncode}\n\n"
        f"STDOUT\n{process.stdout}\nSTDERR\n{process.stderr}"
    )
    (output_dir / "test_report.txt").write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
