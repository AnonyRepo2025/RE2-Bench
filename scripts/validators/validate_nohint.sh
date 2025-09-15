#!/bin/bash

TRIPLES=(
    "../../results/validations/deepseek-coder-33b-instruct_input_prediction_wohint_metadata.json|../../results/validations/deepseek-coder-33b-instruct_input_prediction_wohint.csv|input|../../results/summary/deepseek-coder-33b-instruct_input_prediction_wohint.jsonl"
    "../../results/validations/deepseek-coder-33b-instruct_output_prediction_wohint_metadata.json|../../results/validations/deepseek-coder-33b-instruct_output_prediction_wohint.csv|output|../../results/summary/deepseek-coder-33b-instruct_output_prediction_wohint.jsonl"
    "../../results/validations/o4-mini_input_prediction_wohint_metadata.json|../../results/validations/o4-mini_input_prediction_wohint.csv|input|../../results/summary/o4-mini_input_prediction_wohint.jsonl"
    "../../results/validations/deepseek-reasoner_input_prediction_wohint_metadata.json|../../results/validations/deepseek-reasoner_input_prediction_wohint.csv|input|../../results/summary/deepseek-reasoner_input_prediction_wohint.jsonl"
    "../../results/validations/gemini-pro-1.5_input_prediction_wohint_metadata.json|../../results/validations/gemini-pro-1.5_input_prediction_wohint.csv|input|../../results/summary/gemini-pro-1.5_input_prediction_wohint.jsonl"
    "../../results/validations/gemini-2.5-pro_input_prediction_wohint_metadata.json|../../results/validations/gemini-2.5-pro_input_prediction_wohint.csv|input|../../results/summary/gemini-2.5-pro_input_prediction_wohint.jsonl"
    "../../results/validations/gpt-4.1_input_prediction_wohint_metadata.json|../../results/validations/gpt-4.1_input_prediction_wohint.csv|input|../../results/summary/gpt-4.1_input_prediction_wohint.jsonl"
     "../../results/validations/o4-mini_output_prediction_wohint_metadata.json|../../results/validations/o4-mini_output_prediction_wohint.csv|output|../../results/summary/o4-mini_output_prediction_wohint.jsonl"
    "../../results/validations/deepseek-reasoner_output_prediction_wohint_metadata.json|../../results/validations/deepseek-reasoner_output_prediction_wohint.csv|output|../../results/summary/deepseek-reasoner_output_prediction_wohint.jsonl"
    "../../results/validations/gemini-pro-1.5_output_prediction_wohint_metadata.json|../../results/validations/gemini-pro-1.5_output_prediction_wohint.csv|output|../../results/summary/gemini-pro-1.5_output_prediction_wohint.jsonl"
    "../../results/validations/gemini-2.5-pro_output_prediction_wohint_metadata.json|../../results/validations/gemini-2.5-pro_output_prediction_wohint.csv|output|../../results/summary/gemini-2.5-pro_output_prediction_wohint.jsonl"
    "../../results/validations/gpt-4.1_output_prediction_wohint_metadata.json|../../results/validations/gpt-4.1_output_prediction_wohint.csv|output|../../results/summary/gpt-4.1_output_prediction_wohint.jsonl"
)

PYTHON_SCRIPT="./validate_nohint_results.py"

for triple in "${TRIPLES[@]}"; do
    IFS='|' read -r metadata output task path <<< "$triple"

    echo "Running: python $PYTHON_SCRIPT --summary $path --output_path $output --task $task --metadata_path $metadata"
    python "$PYTHON_SCRIPT" --summary "$path" --output_path "$output" --task "$task" --metadata_path "$metadata"
done