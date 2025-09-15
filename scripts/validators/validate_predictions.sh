#!/bin/bash

TRIPLES=(
    "../../results/validations/deepseek-reasoner_input_metadata.json|../../results/validations/deepseek-reasoner_input.csv|input|../../results/summary/deepseek-reasoner_input_prediction.jsonl"
    "../../results/validations/deepseek-reasoner_output_metadata.json|../../results/validations/deepseek-reasoner_output.csv|output|../../results/summary/deepseek-reasoner_output_prediction.jsonl"
    "../../results/validations/deepseek-coder-33b-instruct_output_metadata.json|../../results/validations/deepseek-coder-33b-instruct_output.csv|output|../../results/summary/deepseek-coder-33b-instruct_output_prediction.jsonl"
    "../../results/validations/deepseek-coder-33b-instruct_input_metadata.json|../../results/validations/deepseek-coder-33b-instruct_input.csv|input|../../results/summary/deepseek-coder-33b-instruct_input_prediction.jsonl"
    "../../results/validations/gemini-1.5-pro_input_metadata.json|../../results/validations/gemini-1.5-pro_input.csv|input|../../results/summary/gemini-1.5-pro_input_prediction.jsonl"
    "../../results/validations/gemini-1.5-pro_output_metadata.json|../../results/validations/gemini-1.5-pro_output.csv|output|../../results/summary/gemini-1.5-pro_output_prediction.jsonl"
    "../../results/validations/gemini-2.5-pro_input_metadata.json|../../results/validations/gemini-2.5-pro_input.csv|input|../../results/summary/gemini-2.5-pro_input_prediction.jsonl"
    "../../results/validations/gemini-2.5-pro_output_metadata.json|../../results/validations/gemini-2.5-pro_output.csv|output|../../results/summary/gemini-2.5-pro_output_prediction.jsonl"
    "../../results/validations/gpt-4.1_input_metadata.json|../../results/validations/gpt-4.1_input.csv|input|../../results/summary/gpt-4.1_input_prediction.jsonl"
    "../../results/validations/gpt-4.1_output_metadata.json|../../results/validations/gpt-4.1_output.csv|output|../../results/summary/gpt-4.1_output_prediction.jsonl"
    "../../results/validations/o4-mini_input_metadata.json|../../results/validations/o4-mini_input.csv|input|../../results/summary/o4-mini_input_prediction.jsonl"
    "../../results/validations/o4-mini_output_metadata.json|../../results/validations/o4-mini_output.csv|output|../../results/summary/o4-mini_output_prediction.jsonl"
)

PYTHON_SCRIPT="./validate_non_swe_bench_results.py"

for triple in "${TRIPLES[@]}"; do
    IFS='|' read -r metadata output task path <<< "$triple"

    echo "Running: python $PYTHON_SCRIPT --summary $path --output_path $output --task $task --metadata_path $metadata"
    python "$PYTHON_SCRIPT" --summary "$path" --output_path "$output" --task "$task" --metadata_path "$metadata"
done