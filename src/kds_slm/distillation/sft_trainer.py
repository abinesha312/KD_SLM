"""LoRA SFT distillation trainer for Qwen student."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from kds_slm.config import AppConfig
from kds_slm.data.formatters import build_chat_messages
from kds_slm.distillation.teacher_labeling import load_teacher_labels
from kds_slm.logging_utils import get_logger
from kds_slm.models.qwen_student import QwenStudent

logger = get_logger(__name__)


class TrainingLogCallback:
    """Collect training metrics for visualization."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.history: list[dict] = []

    def on_log(self, logs: dict) -> None:
        entry = {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()}
        self.history.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


def _format_sft_example(row: dict, system_prompt: str) -> str:
    messages = build_chat_messages(system_prompt, row["prompt"])
    messages.append({"role": "assistant", "content": row["teacher_response"]})
    return messages


def build_sft_dataset(cfg: AppConfig, split: str) -> Dataset:
    df = load_teacher_labels(cfg, split)

    def to_messages(row):
        return {
            "messages": _format_sft_example(row, cfg.system_prompt),
        }

    records = [to_messages(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


def train_distilled_student(cfg: AppConfig) -> Path:
    train_ds = build_sft_dataset(cfg, "train")
    val_ds = build_sft_dataset(cfg, "val")

    cache_path = cfg.resolve(cfg.cache_dir) / "qwen_student"
    model_path = str(cache_path if cache_path.exists() else cfg.student_id)

    student = QwenStudent(cfg, model_path=model_path)
    student.load(model_path)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    student.model = get_peft_model(student.model, lora_config)
    student.model.print_trainable_parameters()

    checkpoint_dir = cfg.resolve(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.resolve(cfg.training_logs_path)
    log_callback = TrainingLogCallback(log_path)

    def formatting_func(example):
        text = student.processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return text

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=student.model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=student.processor,
        formatting_func=formatting_func,
    )

    original_log = trainer.log

    def patched_log(logs: dict) -> None:
        original_log(logs)
        log_callback.on_log(logs)

    trainer.log = patched_log  # type: ignore[method-assign]

    logger.info("Starting distillation training on %d examples", len(train_ds))
    trainer.train()

    adapter_dir = checkpoint_dir / "lora_adapter"
    merged_dir = checkpoint_dir / "merged"
    trainer.model.save_pretrained(str(adapter_dir))
    student.processor.save_pretrained(str(adapter_dir))

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    student.processor.save_pretrained(str(merged_dir))

    logger.info("Saved LoRA adapter to %s and merged model to %s", adapter_dir, merged_dir)
    return checkpoint_dir
