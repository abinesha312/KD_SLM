"""Aggregate comparison results and export summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kds_slm.config import AppConfig
from kds_slm.evaluation.metrics import aggregate_metrics, compute_sample_metrics
from kds_slm.logging_utils import get_logger

logger = get_logger(__name__)


def run_evaluation(cfg: AppConfig) -> tuple[pd.DataFrame, dict]:
    results_dir = cfg.resolve(cfg.qa_results_dir)
    comparison_path = results_dir / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison results not found: {comparison_path}")

    comparison_df = pd.read_csv(comparison_path)
    metrics_df = compute_sample_metrics(comparison_df)
    summary = aggregate_metrics(metrics_df)

    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved metrics to %s", results_dir / "metrics.csv")
    return metrics_df, summary


def render_markdown_report(summary: dict, output_path: Path) -> None:
    lines = [
        "# Medical KD Comparison Report",
        "",
        "## Per-Model Averages",
        "",
        "| Model | ROUGE-L (ref) | BLEU (ref) | BERTScore (ref) | Latency (s) | Tokens/s | VRAM (GB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("per_model", []):
        lines.append(
            f"| {row['model_name']} | "
            f"{row['rouge_l_vs_reference']:.4f} | "
            f"{row['bleu_vs_reference']:.4f} | "
            f"{row['bertscore_vs_reference']:.4f} | "
            f"{row['latency_sec']:.3f} | "
            f"{row['tokens_per_sec']:.1f} | "
            f"{row['peak_vram_gb']:.2f} |"
        )

    lift = summary.get("distilled_minus_base", {})
    if lift:
        lines.extend(
            [
                "",
                "## Distillation Lift (Distilled − Base)",
                "",
            ]
        )
        for metric, value in lift.items():
            lines.append(f"- **{metric}**: {value:+.4f}")

    lines.extend(
        [
            "",
            "> Disclaimer: Model outputs are for research/demo purposes only and are not medical advice.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
