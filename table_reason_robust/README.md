# Table-Reason Robustness Construction

This framework constructs answer-preserving robustness attacks for structured
data analysis tasks. It perturbs only the input table files, never the question.

## Pipeline

1. Selection: choose applicable attack operators from the enabled list.
2. Construct: an LLM constructor uses Python tools to copy and perturb files.
3. Framework integrity validation: deterministic checks verify table structure,
   values, types, formulas, number formats, CSV lexical preservation, and
   operator-specific contracts.
4. Blind judge validation: an independent LLM judge receives isolated original
   and attacked snapshots and verifies that the task remains answerable and the
   normalized answer is preserved.

## Run

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
python src/construct_dataset.py --config config.yaml --limit 2 --num-workers 1
```

For DABench-style CSV tasks:

```bash
cp config_dabench.example.yaml config.yaml
python src/construct_dataset.py --config config.yaml --limit 2 --num-workers 1
```

## Main Files

- `src/construct_dataset.py`: main dataset-construction entry.
- `src/operators.py`: robustness operator registry.
- `src/attack_integrity.py`: deterministic framework integrity validation.
- `src/agent.py`: OpenAI-compatible LLM agent with isolated Python tool calls.
- `prompts/robustness/`: selection, construction, and blind judge prompts.
- `tests/`: unit and integration tests for operators and validation.

## Notes

Use environment variables for credentials. Do not commit `config.yaml`,
generated `outputs/`, or workspaces.
