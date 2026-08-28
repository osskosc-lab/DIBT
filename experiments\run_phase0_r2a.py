from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dibt.phase0_r2a import run_phase0_r2a  # noqa: E402
from dibt.r2a_config import load_phase0_r2a_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_r2a.yaml")
    parser.add_argument(
        "--preregistration",
        default="configs/preregistration_phase0_r2a.yaml",
    )
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument("--preregistration-sha", required=True)
    args = parser.parse_args()
    cfg = load_phase0_r2a_config(args.config)
    _, summary = run_phase0_r2a(
        cfg,
        config_path=args.config,
        preregistration_path=args.preregistration,
        implementation_commit_sha=args.implementation_sha,
        preregistration_commit_sha=args.preregistration_sha,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
