"""Evaluation metrics for medical QA comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

MODEL_LABELS = {
    "teacher": "Gemma E2B (Teacher)",
    "base_student": "Qwen Omni 7B (Base)",
    "distilled_student": "Qwen Omni 7B (Distilled)",
}


@dataclass
class SampleMetrics:
    id: str
    model_name: str
    rouge_l_vs_reference: float
    bleu_vs_reference: float
    rouge_l_vs_teacher: float
    bertscore_vs_reference: float
    latency_sec: float
    tokens_per_sec: float
    peak_vram_gb: float


def _rouge_l(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def _bleu(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    bleu = BLEU(effective_order=True)
    return bleu.sentence_score(prediction, [reference]).score / 100.0


def _batch_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not valid_pairs:
        return [0.0] * len(predictions)
    preds, refs = zip(*valid_pairs)
    _, _, f1 = bert_score_fn(list(preds), list(refs), lang="en", verbose=False)
    scores = [float(x) for x in f1.tolist()]
    if len(scores) == len(predictions):
        return scores
    full = []
    score_idx = 0
    for p, r in zip(predictions, references):
        if r.strip():
            full.append(scores[score_idx])
            score_idx += 1
        else:
            full.append(0.0)
    return full


def compute_sample_metrics(comparison_df: pd.DataFrame) -> pd.DataFrame:
    teacher_map = (
        comparison_df[comparison_df["model_name"] == "teacher"]
        .set_index("id")["response"]
        .to_dict()
    )

    predictions = comparison_df["response"].tolist()
    references = comparison_df["reference_answer"].tolist()
    bert_scores = _batch_bertscore(predictions, references)

    rows: list[dict] = []
    for pos, (_, row) in enumerate(comparison_df.iterrows()):
        teacher_resp = teacher_map.get(row["id"], "")
        rows.append(
            {
                "id": row["id"],
                "model_name": row["model_name"],
                "rouge_l_vs_reference": _rouge_l(row["response"], row["reference_answer"]),
                "bleu_vs_reference": _bleu(row["response"], row["reference_answer"]),
                "rouge_l_vs_teacher": _rouge_l(row["response"], teacher_resp),
                "bertscore_vs_reference": bert_scores[pos],
                "latency_sec": row["latency_sec"],
                "tokens_per_sec": row["tokens_per_sec"],
                "peak_vram_gb": row["peak_vram_gb"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame) -> dict:
    numeric_cols = [
        "rouge_l_vs_reference",
        "bleu_vs_reference",
        "rouge_l_vs_teacher",
        "bertscore_vs_reference",
        "latency_sec",
        "tokens_per_sec",
        "peak_vram_gb",
    ]
    grouped = metrics_df.groupby("model_name")[numeric_cols].mean().reset_index()
    summary = grouped.to_dict(orient="records")

    lift: dict[str, dict[str, float]] = {}
    base = grouped[grouped["model_name"] == "base_student"]
    distilled = grouped[grouped["model_name"] == "distilled_student"]
    if not base.empty and not distilled.empty:
        base_row = base.iloc[0]
        dist_row = distilled.iloc[0]
        for col in numeric_cols:
            lift[col] = float(dist_row[col] - base_row[col])

    return {"per_model": summary, "distilled_minus_base": lift}
