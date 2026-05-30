#!/usr/bin/env python3
"""Step 2: Prepare and split the medical conversational dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import load_config
from kds_slm.data.load_medical_dataset import prepare_dataset
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare medical dataset splits")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = prepare_dataset(cfg)
    for name, frame in splits.items():
        logger.info("%s: %d rows", name, len(frame))


if __name__ == "__main__":
    main()
