from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dibt.r1_config import load_phase0_r1_config  # noqa: E402
from dibt.regeneration import run_regeneration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_r1.yaml")
    parser.add_argument("--output-dir", default="results/phase0_r1")
    args = parser.parse_args()
    cfg = load_phase0_r1_config(args.config)
    _, summary = run_regeneration(cfg, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
