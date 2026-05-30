"""Configuration loading and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class AppConfig:
    seed: int = 42
    teacher_id: str = "google/gemma-4-E2B-it"
    student_id: str = "Qwen/Qwen2.5-Omni-7B"
    cache_dir: str = "outputs/models"
    dataset_name: str = "Azmayen/Medical-conversational-data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    processed_dir: str = "outputs/data/processed"
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    do_sample: bool = True
    teacher_labels_dir: str = "outputs/teacher_labels"
    checkpoint_dir: str = "outputs/checkpoints/qwen_distilled"
    training_logs_path: str = "outputs/checkpoints/training_logs.json"
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    qa_results_dir: str = "outputs/qa_results"
    max_samples: int = 100
    figures_dir: str = "outputs/figures"
    hf_token: str | None = None
    root: Path = field(default_factory=_project_root)

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.root / path


def load_config(config_path: str | Path | None = None) -> AppConfig:
    load_dotenv(_project_root() / ".env")
    root = _project_root()
    path = Path(config_path) if config_path else root / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    models = raw.get("models", {})
    dataset = raw.get("dataset", {})
    generation = raw.get("generation", {})
    distillation = raw.get("distillation", {})
    evaluation = raw.get("evaluation", {})

    return AppConfig(
        seed=raw.get("seed", 42),
        teacher_id=models.get("teacher_id", "google/gemma-4-E2B-it"),
        student_id=models.get("student_id", "Qwen/Qwen2.5-Omni-7B"),
        cache_dir=models.get("cache_dir", "outputs/models"),
        dataset_name=dataset.get("name", "Azmayen/Medical-conversational-data"),
        train_ratio=dataset.get("train_ratio", 0.8),
        val_ratio=dataset.get("val_ratio", 0.1),
        test_ratio=dataset.get("test_ratio", 0.1),
        processed_dir=dataset.get("processed_dir", "outputs/data/processed"),
        system_prompt=raw.get("system_prompt", "").strip(),
        temperature=generation.get("temperature", 0.7),
        top_p=generation.get("top_p", 0.9),
        max_new_tokens=generation.get("max_new_tokens", 512),
        do_sample=generation.get("do_sample", True),
        teacher_labels_dir=distillation.get("teacher_labels_dir", "outputs/teacher_labels"),
        checkpoint_dir=distillation.get("checkpoint_dir", "outputs/checkpoints/qwen_distilled"),
        training_logs_path=distillation.get(
            "training_logs_path", "outputs/checkpoints/training_logs.json"
        ),
        learning_rate=distillation.get("learning_rate", 2e-5),
        num_train_epochs=distillation.get("num_train_epochs", 2),
        per_device_train_batch_size=distillation.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=distillation.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=distillation.get("gradient_accumulation_steps", 4),
        lora_r=distillation.get("lora_r", 16),
        lora_alpha=distillation.get("lora_alpha", 32),
        lora_dropout=distillation.get("lora_dropout", 0.05),
        max_seq_length=distillation.get("max_seq_length", 2048),
        qa_results_dir=evaluation.get("qa_results_dir", "outputs/qa_results"),
        max_samples=evaluation.get("max_samples", 100),
        figures_dir=evaluation.get("figures_dir", "outputs/figures"),
        hf_token=os.getenv("HF_TOKEN"),
        root=root,
    )


def ensure_hf_login(cfg: AppConfig) -> None:
    from huggingface_hub import login

    token = cfg.hf_token
    if not token or token == "your_token_here":
        raise ValueError(
            "HF_TOKEN is missing. Copy .env.example to .env and set your Hugging Face token."
        )
    login(token=token)
