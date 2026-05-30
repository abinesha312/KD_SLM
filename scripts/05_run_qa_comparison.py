#!/usr/bin/env python3
"""Step 5: Run QA comparison across teacher, base, and distilled models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kds_slm.config import ensure_hf_login, load_config
from kds_slm.inference.qa_runner import QAComparisonRunner
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model QA outputs")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--query", default=None, help="Single interactive query")
    parser.add_argument("--interactive", action="store_true", help="Prompt for a query")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_hf_login(cfg)

    query = args.query
    if args.interactive:
        query = input("Enter medical question: ").strip()

    runner = QAComparisonRunner(cfg)
    df = runner.run_comparison(
        split=args.split,
        max_samples=args.max_samples or cfg.max_samples,
        single_query=query,
    )
    csv_path, json_path = runner.save_results(df)
    logger.info("Results: %s, %s", csv_path, json_path)


if __name__ == "__main__":
    main()
