"""Teacher pseudo-label generation with resume support."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from kds_slm.config import AppConfig
from kds_slm.data.load_medical_dataset import load_split
from kds_slm.logging_utils import get_logger
from kds_slm.models.gemma_teacher import GemmaTeacher

logger = get_logger(__name__)


def _is_low_quality(text: str) -> bool:
    if not text or len(text.split()) < 5:
        return True
    words = text.lower().split()
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return True
    return False


def _load_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ids.add(row["id"])
    return ids


def generate_teacher_labels(
    cfg: AppConfig,
    split: str = "train",
    max_samples: int | None = None,
    teacher: GemmaTeacher | None = None,
) -> Path:
    split_dir = cfg.resolve(cfg.processed_dir)
    df = load_split(split_dir, split)
    if max_samples is not None:
        df = df.head(max_samples)

    output_dir = cfg.resolve(cfg.teacher_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    existing_ids = _load_existing_ids(output_path)
    logger.info("Resuming teacher labeling: %d rows already cached", len(existing_ids))

    own_teacher = teacher is None
    if own_teacher:
        teacher = GemmaTeacher(cfg)
        cache_path = cfg.resolve(cfg.cache_dir) / "gemma_teacher"
        model_path = cache_path if cache_path.exists() else cfg.teacher_id
        teacher.load(model_path)

    rows_written = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Teacher labels ({split})"):
            row_id = row["id"]
            if row_id in existing_ids:
                continue

            result = teacher.generate(row["prompt"])
            if _is_low_quality(result.text):
                logger.warning("Skipping low-quality teacher output for id=%s", row_id)
                continue

            record = {
                "id": row_id,
                "prompt": row["prompt"],
                "reference_answer": row.get("reference_answer", ""),
                "teacher_response": result.text,
                "gen_time_sec": result.latency_sec,
                "token_count": result.token_count,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            rows_written += 1

    logger.info("Wrote %d new teacher labels to %s", rows_written, output_path)
    return output_path


def load_teacher_labels(cfg: AppConfig, split: str) -> pd.DataFrame:
    path = cfg.resolve(cfg.teacher_labels_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Teacher labels not found: {path}")
    return pd.read_json(path, lines=True)


def iter_teacher_label_splits(cfg: AppConfig, splits: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {split: load_teacher_labels(cfg, split) for split in splits}
