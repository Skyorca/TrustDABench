# Table Reason Robustness Eval

This project runs agentic LLM inference on robustness attack datasets.
Inference calls the model, executes Python tools, and saves final answers and
complete execution traces. A separate judging stage checks answer correctness,
using an LLM judge or the DABENCH closed-form scorer.

Example input and output paths, relative to this project root. Extract the
published dataset under `../data/TrustDABench-Data-Full/`:

```yaml
dataset_path: "../data/TrustDABench-Data-Full/robustness/aida_qa/attack_dataset.jsonl"
output_path: "outputs/aida_qa/predictions.jsonl"
```

The inference result is a JSONL file. Each row records one sample attempt,
including the question, gold answer, final model response, conversation history,
Python tool trace, run status, and elapsed time.

## Running Inference

Run commands from this project's root directory. Paths below are portable examples;
adjust them to your dataset layout. Inference data/output paths and judge
`input_paths` are relative to the current working directory.

Configure the evaluated model in `configs/config.example.yaml` (or your own
configuration file):

```yaml
model:
  base_url: "https://api.example.com/v1"
  api_key: "${OPENAI_API_KEY}"
  name: "your-model-name"
```

### Gemini Native API

When `model.name` contains `gemini` (case-insensitive), inference uses Gemini's
native `generateContent` protocol. The base URL must explicitly end in
`/v1beta`:

```yaml
model:
  base_url: "https://api.tokenops.ai/v1beta"
  api_key: "${TOKENOPS_API_KEY}"
  name: "gemini-2.5-pro"
  max_tokens: 8192
  temperature: 0.2
  top_p: 1.0
```

The native branch maps `execute_code` to Gemini function calling and preserves
native model parts across tool rounds. Empty responses are retried and then
recorded as `api_failed`, never as successful empty answers. This branch applies
only to the evaluated model, not the judge. Use a new `output_path` when
rerunning old invalid Gemini results.

Run inference:

```bash
python src/run_inference.py --config configs/config.example.yaml
```

Run a small batch:

```bash
python src/run_inference.py --config configs/config.example.yaml --limit 3 --num-workers 1
```

Check dataset loading, file copying, and whether prompts expose absolute paths
without calling the model:

```bash
python src/run_inference.py --config configs/config.example.yaml --limit 3 --dry-run
```

## Prompt Organization

The system prompt is stored in `prompts/system_prompt.txt`:

```text
You are a data analyst...
```

The user prompt is stored in `prompts/user_prompt.txt`. At runtime, the template
is populated with the question and the filenames provided for the current sample.

The model does not see:

- Host-specific absolute paths.
- `file_root`.
- The gold answer.
- The attack type.

It sees filenames in its current working directory, for example:

```text
You are given the following local data files in the current Python working directory:

- page1.xlsx

Question:
...
```

The system prompt requests the following final-answer format:

```text
<final_answer>
...
</final_answer>
```

Text inside these tags is extracted into `prediction.final_answer`. If the tags
are missing, the full last assistant response is used instead, and
`prediction.final_answer_source` is set to `fallback_raw_content`. The original
last response is always retained in `prediction.raw_final_answer`.

## Tool Calling

Non-Gemini models use the standard OpenAI Chat Completions format with
`tools` / `tool_calls`:

```json
{
  "type": "function",
  "function": {
    "name": "execute_code",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string"}
      },
      "required": ["code"]
    }
  }
}
```

Code runs in a separate persistent Python sandbox for each sample. Tool calls
within a sample share variable state; different samples are isolated. Python
errors are returned to the model as tool responses instead of immediately
terminating the sample.

The OpenAI-compatible inference branch does not send `temperature`. The Gemini
native branch supports the configuration parameters shown above.

## Prediction JSONL Fields

Each row contains the following main fields:

- `id`: Attack sample ID.
- `source_id`: Original sample ID.
- `attack_type`: Attack type, retained for offline analysis and excluded from prompts.
- `question`: Question presented to the model.
- `gold_answer`: Reference answer for the task. Retained in the output and excluded from prompts.
- `input_file` / `file_root`: File location metadata from the attack dataset, excluded from prompts.
- `provided_files`: Filenames copied into the sample workspace and made available to the model.
- `prompt`: The actual system and user prompts sent to the model.
- `prediction.raw_final_answer`: Original text of the last assistant response.
- `prediction.final_answer`: Text extracted from `<final_answer>`, or the full last response when tags are absent.
- `prediction.final_answer_source`: `final_answer_tag` or `fallback_raw_content`.
- `prediction.messages`: Conversation history in the framework's OpenAI-style message representation.
- `prediction.tool_events`: Code and output for each Python tool call.
- `status`: `success`, `run_failed`, `api_failed`, or `workspace_failed`.
- `elapsed_sec`: Sample runtime in seconds.

## Resuming Inference

At startup, the runner reads the prediction JSONL. An ID is complete only if it
has at least one record with `status == success` and a nonempty final answer.
Records with `api_failed`, `run_failed`, `workspace_failed`, or empty answers
are queued again on the next run. Each completed attempt is appended as a new
line, so interrupted runs can be resumed without removing failed attempts.

Multiple rows for the same ID are expected after retries and serve as an audit
history. The judge selects one canonical record using the "last valid success
wins" rule. The startup log's `completed_successfully` count includes only
successful tasks with nonempty answers.

To force a successful sample to run again, remove all successful records for
that ID from the prediction file before rerunning.

## Relationship to dsbench

This project reuses dsbench's persistent process-level Python sandbox design
and `sandboxv2.py` implementation. Each sample has its own Python subprocess,
retains variable state across tool calls, and supports forced timeout cleanup.

The following dsbench components are not reused:

- The XML `<tool_call>` protocol.
- Kaggle / `submission.csv` artifact handling.
- Separate result directories containing multiple output files.
- Automatic evaluation scripts.

Inference results are consolidated into a single `predictions.jsonl` file for
table question answering and robustness attack evaluation. Scoring is performed
separately as described below.

## Answer Correctness Judge

Inference collects model answers and traces. A separate judge script evaluates
answer correctness:

```bash
python src/judge_refusal.py --config judge_config.example.yaml
```

The script name `judge_refusal.py` and the default mode name `llm_refusal` are
retained for compatibility. In this project, the LLM judge evaluates
**correctness**, using `correct` / `incorrect` labels and binary scores.

The judge uses its own `judge_config.example.yaml`, separate from the inference
configuration. This allows the evaluated model and the judge to be different.

Specify multiple prediction files in one configuration:

```yaml
input_paths:
  - "outputs/aida_qa/predictions_qwen37max.jsonl"
  - "outputs/aida_qa/predictions_gpt55.jsonl"
```

Each input file produces its own judgment and report files in the same
directory. The model name comes from the prediction records' `model` field,
with characters that are invalid in filenames replaced. If no model name is
available, the filename prefix is `model`:

```text
<model>_judgments.jsonl
<model>_report.json
```

The `--output` and `--report` options are supported only when there is exactly
one input file. When multiple inputs share a directory and model name, run
them separately with distinct output paths to avoid filename collisions.

## DABENCH Closed-Form Scoring

DABENCH uses an official structured answer format, for example:

```text
@mean_fare[34.65]
@significance[significant]
```

These predictions can be scored deterministically without an LLM judge:

```bash
python src/judge_refusal.py --config judge_config_dabench_closed_form.example.yaml
```

Key configuration fields:

```yaml
judge:
  mode: "dabench_closed_form"

dabench:
  labels_path: "../data/DABENCH/da-dev-labels.jsonl"
  id_prefix: "DA_"
```

The DABENCH paths above are examples; relative questions and labels paths are
resolved against the project root.

In this mode, the script does not read the `model` configuration, perform an
LLM preflight check, or call any model API. Scoring follows
`DABENCH/eval_closed_form.py`: extract `@name[value]` fields from the final
answer and compare them with the official `common_answers`. Strings must
match exactly; numbers allow a tolerance of `1e-6`. The task receives
`score = 1` only when every required field is correct; otherwise it receives
`score = 0`.

Robustness samples are matched to original labels through `source_id`.
For example, `DA_109__unit_scale_conversion__001` uses `source_id = DA_109`,
which maps to official label ID `109`. Clean samples without `source_id`
use their own `id`.

Outputs retain `score`, `label`, `is_correct`, `judge_status`, `source_id`,
and `attack_type`, and can be passed directly to
`compute-robust/compute_robust_scores.py`.

Default output names distinguish closed-form scoring from LLM judging:

```text
<model>_dabench_closed_form_judgments.jsonl
<model>_dabench_closed_form_report.json
```

### Duplicate Inference Attempts

Before scheduling judgments, the judge groups records by `id` within each input
prediction file. It selects the last record with `status == success` and a
nonempty final answer. If no valid success exists, it selects the last failed
attempt and marks it as not evaluable under the existing scoring rules.

Deduplication happens before `--limit`, so the limit counts unique tasks rather
than physical JSONL lines. Missing or empty IDs produce an error identifying
the file and line number.

Existing judgment files are not migrated. Use a new output path, or remove old
judgment and report files before rerunning, to avoid retaining duplicate tasks.

The canonical-attempt selection rule applies to both the LLM judge and the
DABENCH closed-form judge.

### LLM Judge Inputs

For each sample, the LLM judge reads three core inputs:

- `question`: The question presented to the evaluated model.
- `gold_answer`: The reference answer for the task.
- `prediction.final_answer`: The model's final answer, with `prediction.raw_final_answer` used as a fallback when necessary.

The judge does not inspect the full tool trace, solve the task again, or
access the original table files. It evaluates the final answer against the
question and reference.

If `status != success` or no usable final answer is available, the judge LLM is
not called. The result is recorded as:

```json
{
  "judge_status": "not_evaluable",
  "score": 0,
  "label": "incorrect",
  "is_correct": false
}
```

These samples contribute zero to the overall score and are also counted
separately in the report's `not_evaluable` field.

### Binary Scoring

- `score = 1`, `label = correct`: The answer is correct.
- `score = 0`, `label = incorrect`: The answer is incorrect, unavailable, or cannot be evaluated successfully.

Robustness evaluation measures whether the model can still answer correctly
after an answer-preserving table attack. It does not use the three-level
refusal rubric from the reliability evaluation framework.

### Judgment Fields

The judgment JSONL contains one record per deduplicated task.

- `judgment_id`: Resume key composed of `input_path`, `id`, and the selected attempt's `input_line`, avoiding collisions across input files.
- `id`: Attack sample ID.
- `source_id`: Original sample ID, used to align attacked and clean results.
- `attack_type`: Robustness operator, such as `row_order_shuffle` or `unit_scale_conversion`.
- `model`: Evaluated model name from the prediction record.
- `run_status`: Inference status, such as `success`, `run_failed`, `api_failed`, or `workspace_failed`.
- `input_path`: Source prediction JSONL.
- `input_line`: Record line number in the input JSONL.
- `question`: Task question.
- `gold_answer`: Reference answer retained from the prediction record.
- `final_answer`: Evaluated model's final response.
- `judge_model`: Judge model or scorer identifier.
- `started_at` / `finished_at`: Judgment start and end timestamps.
- `judge_status`: `success` for a completed judgment; `not_evaluable` when no usable answer exists; `judge_failed` when judging fails.
- `score`: Binary score, `1` or `0`.
- `label`: `correct` or `incorrect`.
- `is_correct`: Boolean correctness result.
- `reason`: Short scoring explanation.
- `raw_judge_response`: Raw judge response, where applicable, retained for debugging.
- `error`: Error details; `null` on success.
- `elapsed_sec`: Judgment runtime in seconds.

DABENCH closed-form results also include scorer-specific evidence, such as
`label_id` and `label_answers`.

### Report Fields

Each report summarizes one prediction file. It is rebuilt from the
corresponding judgment JSONL at the end of each run, so resumed runs reflect
the judgments currently available.

- `total`: Total number of judgments.
- `evaluable`: Number successfully judged, where `judge_status == success`.
- `not_evaluable`: Number not judged successfully, including inference failures, empty answers, and judge failures.
- `mean_score_all`: Mean binary score over all judgments, with failures contributing zero.
- `mean_score_evaluable`: Mean binary score over evaluable judgments.
- `accuracy_all`: Fraction of all judgments that are correct.
- `accuracy_evaluable`: Fraction of evaluable judgments that are correct.
- `correct_count`: Number of correct judgments.
- `incorrect_count`: Total minus the number of correct judgments, including failures.
- `label_counts`: Counts by label.
- `by_attack_type`: The same statistics grouped by attack type.
- `by_model`: The same statistics grouped by evaluated model.

These reports describe attacked-task correctness. For paired robustness
metrics such as ASR and RAD, use `compute-robust/compute_robust_scores.py`
with both clean and attacked judgments.

## Duplicate Clean Judgments in Robustness Scoring

The robustness scoring scripts allow multiple attempts for the same task in
the clean judgment JSONL. Overall ASR/RAD and per-operator statistics use the
same merge rule: for each ID, select the last successful record with a valid
binary score. A later failed record does not replace an earlier successful
judgment.

If an ID never has a successful judgment, retain its last failed attempt and
always treat that clean sample as incorrect. The `unscored-policy` option still
controls how unscored attacked records are handled, but it does not remove
these all-failed clean samples from the denominator.

The summary JSON's `clean_canonicalization` field records physical row counts,
unique task counts, collapsed attempts, and the numbers of selected successful
and failed attempts for auditing resumed runs.
