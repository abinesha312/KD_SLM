#!/usr/bin/env python3
"""Step 4: Distill teacher knowledge into the student via LoRA SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import ensure_hf_login, load_config
from kds_slm.distillation.sft_trainer import train_distilled_student
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train distilled student model")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_hf_login(cfg)

    checkpoint_dir = train_distilled_student(cfg)
    logger.info("Distillation complete. Checkpoints: %s", checkpoint_dir)


if __name__ == "__main__":
    main()
