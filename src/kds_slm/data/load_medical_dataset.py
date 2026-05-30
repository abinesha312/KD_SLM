"""Load and split the medical conversational dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from kds_slm.config import AppConfig, ensure_hf_login
from kds_slm.data.formatters import normalize_row
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def load_raw_dataset(cfg: AppConfig):
    ensure_hf_login(cfg)
    logger.info("Loading dataset %s", cfg.dataset_name)
    return load_dataset(cfg.dataset_name, token=cfg.hf_token)


def normalize_dataset(raw_ds, seed: int) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    split_name = list(raw_ds.keys())[0]
    data = raw_ds[split_name]

    for idx, row in enumerate(data):
        normalized = normalize_row(dict(row), row_id=f"{split_name}_{idx}")
        if normalized:
            records.append(normalized)

    if not records:
        raise ValueError(
            "No rows could be normalized. Inspect dataset schema and update formatters.py."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    logger.info("Normalized %d unique prompts from split '%s'", len(df), split_name)
    return df


def split_dataset(df: pd.DataFrame, cfg: AppConfig) -> dict[str, pd.DataFrame]:
    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio
    test_ratio = cfg.test_ratio
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=cfg.seed, shuffle=True
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=cfg.seed, shuffle=True
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        jsonl_path = output_dir / f"{name}.jsonl"
        parquet_path = output_dir / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in frame.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %s split: %d rows -> %s", name, len(frame), jsonl_path)


def load_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    parquet_path = split_dir / f"{split_name}.parquet"
    jsonl_path = split_dir / f"{split_name}.jsonl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)
    raise FileNotFoundError(f"Split '{split_name}' not found in {split_dir}")


def prepare_dataset(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    raw = load_raw_dataset(cfg)
    df = normalize_dataset(raw, cfg.seed)
    splits = split_dataset(df, cfg)
    out_dir = cfg.resolve(cfg.processed_dir)
    save_splits(splits, out_dir)
    return splits
