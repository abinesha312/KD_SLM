"""Gemma 4 E2B teacher model wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

from kds_slm.config import AppConfig
from kds_slm.data.formatters import build_chat_messages
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    token_count: int
    peak_vram_gb: float


class GemmaTeacher:
    def __init__(self, cfg: AppConfig, model_path: str | Path | None = None):
        self.cfg = cfg
        self.model_id = str(model_path or cfg.teacher_id)
        self.processor = None
        self.model = None

    def download(self) -> Path:
        cache = self.cfg.resolve(self.cfg.cache_dir) / "gemma_teacher"
        cache.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading teacher model %s", self.cfg.teacher_id)
        snapshot_download(
            repo_id=self.cfg.teacher_id,
            local_dir=str(cache),
            token=self.cfg.hf_token,
        )
        return cache

    def load(self, model_path: str | Path | None = None) -> None:
        path = str(model_path or self.cfg.teacher_id)
        logger.info("Loading Gemma teacher from %s", path)
        self.processor = AutoProcessor.from_pretrained(path, token=self.cfg.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            token=self.cfg.hf_token,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, user_prompt: str, system_prompt: str | None = None) -> GenerationResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        system = system_prompt if system_prompt is not None else self.cfg.system_prompt
        messages = build_chat_messages(system, user_prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        new_tokens = outputs[0][input_len:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        return GenerationResult(
            text=response,
            latency_sec=latency,
            token_count=len(new_tokens),
            peak_vram_gb=peak_vram,
        )

    def smoke_test(self) -> GenerationResult:
        return self.generate("What are common symptoms of seasonal allergies?")
