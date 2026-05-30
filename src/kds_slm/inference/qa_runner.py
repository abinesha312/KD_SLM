"""Unified QA runner comparing teacher, base student, and distilled student."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from kds_slm.config import AppConfig
from kds_slm.data.load_medical_dataset import load_split
from kds_slm.logging_utils import get_logger
from kds_slm.models.gemma_teacher import GemmaTeacher
from kds_slm.models.qwen_student import QwenStudent

logger = get_logger(__name__)


@dataclass
class QAResultRow:
    id: str
    prompt: str
    reference_answer: str
    model_name: str
    response: str
    latency_sec: float
    token_count: int
    peak_vram_gb: float
    tokens_per_sec: float


def _unload(*models) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QAComparisonRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.teacher_cache = cfg.resolve(cfg.cache_dir) / "gemma_teacher"
        self.student_cache = cfg.resolve(cfg.cache_dir) / "qwen_student"
        self.distilled_path = cfg.resolve(cfg.checkpoint_dir) / "lora_adapter"

    def _teacher_path(self) -> str:
        return str(self.teacher_cache if self.teacher_cache.exists() else self.cfg.teacher_id)

    def _student_path(self) -> str:
        return str(self.student_cache if self.student_cache.exists() else self.cfg.student_id)

    def run_single_model(
        self,
        model_name: str,
        prompts_df: pd.DataFrame,
    ) -> list[QAResultRow]:
        results: list[QAResultRow] = []

        if model_name == "teacher":
            model = GemmaTeacher(self.cfg)
            model.load(self._teacher_path())
            generate_fn = model.generate
        elif model_name == "base_student":
            model = QwenStudent(self.cfg, model_path=self._student_path())
            model.load(self._student_path())
            generate_fn = model.generate
        elif model_name == "distilled_student":
            adapter = str(self.distilled_path) if self.distilled_path.exists() else None
            model = QwenStudent(self.cfg, model_path=self._student_path(), adapter_path=adapter)
            model.load(self._student_path(), adapter_path=adapter)
            generate_fn = model.generate
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc=model_name):
            gen = generate_fn(row["prompt"])
            tps = gen.token_count / gen.latency_sec if gen.latency_sec > 0 else 0.0
            results.append(
                QAResultRow(
                    id=row["id"],
                    prompt=row["prompt"],
                    reference_answer=row.get("reference_answer", ""),
                    model_name=model_name,
                    response=gen.text,
                    latency_sec=gen.latency_sec,
                    token_count=gen.token_count,
                    peak_vram_gb=gen.peak_vram_gb,
                    tokens_per_sec=tps,
                )
            )

        _unload(model.model, model)
        return results

    def run_comparison(
        self,
        split: str = "test",
        max_samples: int | None = None,
        single_query: str | None = None,
    ) -> pd.DataFrame:
        if single_query:
            prompts_df = pd.DataFrame(
                [{"id": "interactive_0", "prompt": single_query, "reference_answer": ""}]
            )
        else:
            split_dir = self.cfg.resolve(self.cfg.processed_dir)
            prompts_df = load_split(split_dir, split)
            if max_samples is not None:
                prompts_df = prompts_df.head(max_samples)

        all_rows: list[QAResultRow] = []
        for model_name in ("teacher", "base_student", "distilled_student"):
            logger.info("Running QA for model: %s", model_name)
            if model_name == "distilled_student" and not self.distilled_path.exists():
                logger.warning(
                    "Distilled checkpoint not found at %s; skipping distilled_student.",
                    self.distilled_path,
                )
                continue
            all_rows.extend(self.run_single_model(model_name, prompts_df))

        return pd.DataFrame([asdict(r) for r in all_rows])

    def save_results(self, df: pd.DataFrame) -> tuple[Path, Path]:
        out_dir = self.cfg.resolve(self.cfg.qa_results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "comparison.csv"
        json_path = out_dir / "comparison.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        logger.info("Saved comparison results to %s", csv_path)
        return csv_path, json_path
