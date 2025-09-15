# Validators

This directory contains validation scripts for evaluating LLM-generated responses across multiple benchmark projects.

## Prerequisites

Before using these validators, you must install all requirements for every benchmark project you want to validate. We have tested these validators with the following projects:

### Supported Benchmarks

- **ClassEval**: [Installation Instructions](https://github.com/FudanSELab/ClassEval)
- **HumanEval**: [Installation Instructions](https://github.com/openai/human-eval)
- **CruxEval**: [Installation Instructions](https://github.com/facebookresearch/cruxeval)
- **Avatar**: [Installation Instructions](https://github.com/wasiahmad/AVATAR)

Please follow each project's specific installation instructions to set up the required dependencies.

## Usage

### General Validation

To validate all generated responses in `results/summary`, run:

```bash
./validate_predictions.sh
```

You can modify the script to add or remove projects as needed.

### No Hint Validation

To validate LLM-generated responses in the no-hint condition:

```bash
./validate_nohint.sh
```

### No Chain-of-Thought Validation

To validate generated responses without chain-of-thought reasoning:

```bash
./validate_nocot.sh
```

## Configuration

Each bash script requires you to configure a list of 4 parameters:

1. **JSON Output Path**: Path where you want to save the total results (including RS and partial RS)
2. **CSV Output Path**: Path where you want to save detailed results for each case
3. **Task Type**: Specify the task type:
   - `input` - for input prediction tasks
   - `output` - for output prediction tasks
4. **Summary File Path**: Path to the summary file containing the generated responses to validate

## Output Files

The validation scripts generate two types of output files:

### CSV File
Contains detailed validation responses for each case, including:
- RS (Response Score)
- Partial RS
- False Negative indicators

### JSON File
Contains aggregated statistics:
- Number of False Negatives
- False negatives per difficulty level
- RS (Response Score) metrics
- RS per difficulty level
- Partial RS metrics
- Partial RS per difficulty level
- RS and partial RS considering false negatives as correct predictions

## Example Configuration

The bash scripts use an array of triples where each triple contains 4 pipe-separated values:

```bash
TRIPLES=(
    "metadata_output_path|csv_output_path|task_type|summary_file_path"
    "metadata_output_path|csv_output_path|task_type|summary_file_path"
    # ... more entries
)
```

### Parameter Details:
- **metadata_output_path**: Path for JSON metadata file (e.g., `../../results/validations/model_input_prediction_metadata.json`)
- **csv_output_path**: Path for detailed CSV results (e.g., `../../results/validations/model_input_prediction.csv`)
- **task_type**: Either `input` or `output`
- **summary_file_path**: Path to the summary file with generated responses (e.g., `../../results/summary/model_input_prediction.jsonl`)

### Real Example from validate_nocot.sh:
```bash
"../../results/validations/deepseek-coder-33b-instruct_input_prediction_wocot_metadata.json|../../results/validations/deepseek-coder-33b-instruct_input_prediction_wocot.csv|input|../../results/summary/deepseek-coder-33b-instruct_input_prediction_wocot.jsonl"
```
