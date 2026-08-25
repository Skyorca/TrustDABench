# SecurityTableBench Code

This package contains the data-construction code for SecurityTableBench.
It builds reliability and robustness attack samples for structured-data
analysis tasks.

The released code is separated from the released data. Generated datasets,
intermediate workspaces, model logs, API credentials, and local paths are not
included in this code package.

## Components

- `table_reason_reliability/`: constructs unanswerable table-analysis samples.
  The expected model behavior is to refuse with the correct missing/conflicting
  evidence.
- `table_reason_robust/`: constructs answer-preserving perturbations. The
  expected model behavior is to solve the perturbed task and preserve the
  normalized answer.

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

