#!/usr/bin/env python3
"""Step 3: Generate teacher pseudo-labels for distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import ensure_hf_login, load_config
from kds_slm.distillation.teacher_labeling import generate_teacher_labels
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher pseudo-labels")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "all"],
        help="Dataset split to label",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_hf_login(cfg)

    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        path = generate_teacher_labels(cfg, split=split, max_samples=args.max_samples)
        logger.info("Teacher labels saved: %s", path)


if __name__ == "__main__":
    main()
