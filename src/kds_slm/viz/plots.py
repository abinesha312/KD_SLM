"""Visualization utilities for KD comparison results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kds_slm.evaluation.metrics import MODEL_LABELS

sns.set_theme(style="whitegrid")


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def plot_quality_bar(summary: dict, output_path: Path) -> None:
    df = pd.DataFrame(summary["per_model"])
    df["label"] = df["model_name"].map(_label)
    metrics = ["rouge_l_vs_reference", "bertscore_vs_reference"]
    melted = df.melt(id_vars="label", value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="label", y="score", hue="metric", ax=ax)
    ax.set_title("Quality Metrics by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_efficiency_bar(summary: dict, output_path: Path) -> None:
    df = pd.DataFrame(summary["per_model"])
    df["label"] = df["model_name"].map(_label)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=df, x="label", y="latency_sec", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Mean Latency (seconds)")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=df, x="label", y="peak_vram_gb", ax=axes[1], palette="Greens_d")
    axes[1].set_title("Mean Peak VRAM (GB)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_quality_scatter(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df.copy()
    plot_df["label"] = plot_df["model_name"].map(_label)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="latency_sec",
        y="rouge_l_vs_reference",
        hue="label",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Latency vs ROUGE-L (Reference)")
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("ROUGE-L vs Reference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(log_path: Path, output_path: Path) -> None:
    if not log_path.exists():
        return
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    train_logs = [h for h in history if "loss" in h]
    eval_logs = [h for h in history if "eval_loss" in h]
    if not train_logs and not eval_logs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_logs:
        steps = list(range(len(train_logs)))
        losses = [h["loss"] for h in train_logs]
        ax.plot(steps, losses, label="train_loss")
    if eval_logs:
        steps = list(range(len(eval_logs)))
        losses = [h["eval_loss"] for h in eval_logs]
        ax.plot(steps, losses, label="eval_loss")

    ax.set_title("Distillation Training Curves")
    ax.set_xlabel("Log Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distillation_lift(summary: dict, output_path: Path) -> None:
    lift = summary.get("distilled_minus_base", {})
    if not lift:
        return
    quality_metrics = ["rouge_l_vs_reference", "bleu_vs_reference", "bertscore_vs_reference"]
    data = [{"metric": m, "delta": lift[m]} for m in quality_metrics if m in lift]
    if not data:
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="metric", y="delta", ax=ax, palette="Purples_d")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Distillation Lift (Distilled − Base)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_all_figures(
    summary: dict,
    metrics_df: pd.DataFrame,
    figures_dir: Path,
    training_log_path: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures_dir / "quality_bar.png",
        figures_dir / "efficiency_bar.png",
        figures_dir / "latency_vs_rouge_scatter.png",
        figures_dir / "training_curves.png",
        figures_dir / "distillation_lift.png",
    ]
    plot_quality_bar(summary, outputs[0])
    plot_efficiency_bar(summary, outputs[1])
    plot_latency_quality_scatter(metrics_df, outputs[2])
    plot_training_curves(training_log_path, outputs[3])
    plot_distillation_lift(summary, outputs[4])
    return [p for p in outputs if p.exists()]
