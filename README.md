# TrustDABench Code

This repository contains the data-construction and evaluation code for
TrustDABench. It builds and evaluates reliability and robustness attack
samples for structured-data analysis tasks.

Data:  https://modelscope.cn/datasets/Skyorca/TrustDABench

## Components

- `table_reason_reliability/`: constructs unanswerable table-analysis samples.
  The expected model behavior is to refuse with the correct missing/conflicting
  evidence.
- `table_reason_robust/`: constructs answer-preserving perturbations. The
  expected model behavior is to solve the perturbed task and preserve the
  normalized answer.
- `table_reason_reliability_eval/`: runs model inference on reliability
  samples, judges refusal behavior, and computes reliability summaries.
- `table_reason_robust_eval/`: runs model inference on robustness samples,
  judges answer correctness, and computes ASR/RAD and per-operator summaries.

Both frameworks follow the same high-level pipeline:

1. Selection: choose applicable attack operators for an original sample.
2. Construct: use an LLM agent and Python tools to create perturbed table files.
3. Validate: check attack validity with deterministic table rules and LLM-based
   semantic validation.

## Quick Start

```bash
cd table_reason_robust
pip install -r requirements.txt
cp config.example.yaml config.yaml
python src/construct_dataset.py --config config.yaml --limit 2 --num-workers 1
```

Configure model credentials via environment variables such as
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY`.

## Evaluating Models

Extract the published full dataset under `data/TrustDABench-Data-Full/`.
For DABENCH closed-form scoring, place the official labels under
`data/DABENCH/`. Paths in the example configurations are relative to the
evaluation project directory.

```bash
cd table_reason_robust_eval
pip install -r requirements.txt
python src/run_inference.py --config configs/config.example.yaml --limit 3
python src/judge_refusal.py --config judge_config.example.yaml
```

Use `table_reason_reliability_eval/` and its corresponding example
configurations for refusal evaluation. For DABENCH robustness, use
`judge_config_dabench_closed_form.example.yaml` for deterministic scoring.
See each evaluation directory's README for fields, resuming, and metrics.

