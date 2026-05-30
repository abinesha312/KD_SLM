# Medical Knowledge Distillation Demo Pipeline

Modular, reproducible pipeline demonstrating **domain-specialized knowledge distillation** on medical conversational data. A Gemma 4 E2B teacher generates pseudo-labels; a Qwen2.5-Omni-7B student is fine-tuned via LoRA SFT; all three variants (teacher, base student, distilled student) are compared on identical medical queries with metrics and charts.

> **Disclaimer:** Outputs are for research and demonstration only. They are **not** medical advice and must not be used for diagnosis or treatment.

## Architecture

| Role | Model | Hugging Face ID |
|------|-------|-----------------|
| Teacher | Gemma 4 E2B (instruction-tuned) | `google/gemma-4-E2B-it` |
| Student (base + distilled) | Qwen2.5-Omni-7B | `Qwen/Qwen2.5-Omni-7B` |
| Dataset | Medical QA / consultation data | [`HoangHa/medical-data`](https://huggingface.co/datasets/HoangHa/medical-data) (`RandomQA` subset by default) |

Distillation uses **response-level** pseudo-labeling (cross-architecture; no shared tokenizer required).

### Dataset subsets

[`HoangHa/medical-data`](https://huggingface.co/datasets/HoangHa/medical-data) includes four subsets:

| Subset | Rows | Best for |
|--------|------|----------|
| `RandomQA` | ~67k | **QA distillation** (question + answer pairs) — default |
| `english` | ~109k | Multi-turn English clinical conversations |
| `vietnamese` | ~58k | Multi-turn Vietnamese clinical conversations |
| `RandomQuestion` | ~61k | Question-only prompts (no reference answers) |

Change subset in [`config/default.yaml`](config/default.yaml) or via CLI:

```bash
python scripts/02_prepare_dataset.py --subset english --max-rows 5000
```

## Quick start (local)

```bash
# 1. Clone and install
cd 56_KDS_SLM
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .

# 2. Configure Hugging Face token (required for model downloads)
copy .env.example .env
# Edit .env and set HF_TOKEN=hf_...

# 3. Run pipeline
python scripts/01_download_models.py
python scripts/02_prepare_dataset.py
python scripts/03_generate_teacher_outputs.py --split all
python scripts/04_distill_train.py
python scripts/05_run_qa_comparison.py --split test --max-samples 100
python scripts/06_visualize_results.py
```

## Google Colab (H100)

Open [`notebooks/medical_kd_colab.ipynb`](notebooks/medical_kd_colab.ipynb), set your `HF_TOKEN`, and run all cells. Expected runtime on H100: ~2–4 hours end-to-end depending on dataset size.

## Pipeline steps

| Script | Purpose |
|--------|---------|
| `01_download_models.py` | Download teacher & student to `outputs/models/` |
| `02_prepare_dataset.py` | Normalize, split (80/10/10), save to `outputs/data/processed/` |
| `03_generate_teacher_outputs.py` | Teacher pseudo-labels → `outputs/teacher_labels/` |
| `04_distill_train.py` | LoRA SFT distillation → `outputs/checkpoints/qwen_distilled/` |
| `05_run_qa_comparison.py` | 3-model QA comparison → `outputs/qa_results/` |
| `06_visualize_results.py` | Metrics + charts → `outputs/figures/` |

## Outputs

```
outputs/
├── models/                  # Downloaded checkpoints
├── data/processed/          # train/val/test splits
├── teacher_labels/          # Pseudo-label JSONL
├── checkpoints/qwen_distilled/
│   ├── lora_adapter/
│   ├── merged/
│   └── training_logs.json
├── qa_results/
│   ├── comparison.csv
│   ├── metrics.csv
│   ├── summary.json
│   └── report.md
└── figures/
    ├── quality_bar.png
    ├── efficiency_bar.png
    ├── latency_vs_rouge_scatter.png
    ├── training_curves.png
    └── distillation_lift.png
```

## Configuration

Edit [`config/default.yaml`](config/default.yaml) for model IDs, generation params, LoRA/training hyperparameters, and paths.

## Interactive QA

```bash
python scripts/05_run_qa_comparison.py --interactive
python scripts/05_run_qa_comparison.py --query "What is hypertension?"
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (Colab H100 or 24GB+ VRAM)
- Hugging Face account (for model downloads; dataset is public)

## License

Code: project defaults. Models and dataset subject to their respective Hugging Face licenses.
