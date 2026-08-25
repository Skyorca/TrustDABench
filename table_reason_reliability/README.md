# Table-Reason Reliability Construction

This framework constructs unanswerable structured-data analysis samples. The
expected model behavior is to refuse to answer and identify the missing,
conflicting, or ambiguous evidence.

## Pipeline

1. Selection: choose applicable reliability attack operators.
2. Construct: an LLM constructor perturbs only the table/file package.
3. Validate: a validator checks that the original task was answerable, the
   attacked task is not uniquely answerable, and the refusal reason matches the
   constructed evidence.

## Run

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
python src/construct_dataset.py --config config.yaml --limit 2 --num-workers 1
```

To rebuild the public JSONL from accepted samples:

```bash
python src/construct_dataset.py --config config.yaml --rebuild-only
```

## Main Files

- `src/construct_dataset.py`: main reliability sample-construction entry.
- `src/operators.py`: reliability operator registry.
- `src/dataset.py`: dataset loading and table profiling.
- `src/agent.py`: OpenAI-compatible LLM agent with isolated Python tools.
- `src/validators.py`: payload and validation result checks.
- `prompts/reliability/`: selection, construction, and unanswerability prompts.

## Notes

The framework expects table-analysis datasets with a question, reference answer,
and one or more input files. Use environment variables for credentials. Do not
commit `config.yaml`, generated outputs, or workspaces.
