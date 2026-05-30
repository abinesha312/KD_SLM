#!/usr/bin/env python3
"""Step 1: Download teacher and student models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import ensure_hf_login, load_config
from kds_slm.logging_utils import get_logger
from kds_slm.models.gemma_teacher import GemmaTeacher
from kds_slm.models.qwen_student import QwenStudent

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_hf_login(cfg)

    teacher = GemmaTeacher(cfg)
    student = QwenStudent(cfg)

    teacher_path = teacher.download()
    student_path = student.download()
    logger.info("Teacher cached at %s", teacher_path)
    logger.info("Student cached at %s", student_path)

    if not args.skip_smoke_test:
        logger.info("Running teacher smoke test...")
        teacher.load(teacher_path)
        t_result = teacher.smoke_test()
        logger.info("Teacher response preview: %s", t_result.text[:200])

        logger.info("Running student smoke test...")
        student.load(student_path)
        s_result = student.smoke_test()
        logger.info("Student response preview: %s", s_result.text[:200])


if __name__ == "__main__":
    main()
