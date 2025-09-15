
<p align="center">
    <a href="./false_negative_examples.md">⚠️ Additional False Negatives in Input Prediction</a>
    <a href="/failure_cases.md">🔥Additional Reasoning Failure Cases</a> 
</p>

## Installation
Run the following commands to create the environment:
```
conda create -n re2-bench python=3.9
pip install -r requirements.txt
conda activate re2-bench
```


## Reproducing Results
<!-- ### Extracting inputs/outputs from original benchmarks
For Avatar, CruxEval, HumanEval, and ClassEval
```
cd scripts
bash run_code_instrumentation_generic.sh
``` -->

### Difficulty Level Categorization
```
cd scripts
python categorization/label_sample.py --p1 0.25 --p4 075
```

### Main Experiemnts
#### Prompting LLMs:

Model_ID: {`deepseek/deepseek-r1-0528`, `openai/o4-mini`, `google/gemini-2.5-pro`, `google/gemini-pro-1.5`, `openai/gpt-4.1`, `deepseek-ai/deepseek-coder-33b-instruct`}

TASK: {`input_prediction`, `output_prediction`}
```
cd scripts
python prompting/prompt_models.py --model {MODEL_ID} --task {TASK}
```

#### Parsing results:
```
python prompting/parse_results.py --model {MODEL_ID} --task {TASK}
```

### Ablation Study:
TASK: {`input_prediction_wocot`(input prediction without CoT), `output_prediction_wocot` (output prediction without CoT), `input_prediction_wohint`(input prediction without structual hint), `output_prediction_wohint` (output prediction without structual hint)}
```
python prompting/parse_results.py --model {MODEL_ID} --task {TASK}
```
Results can be found `/results/summary`

### Validation on the results

####  General Validation

To validate all generated responses in `results/summary`, run:

```bash
cd scripts/validators
./validate_predictions.sh
```

You can modify the script to add or remove projects as needed.

#### No Hint Validation

To validate LLM-generated responses in the no-hint condition:

```bash
cd scripts/validators
./validate_nohint.sh
```

#### No Chain-of-Thought Validation

To validate generated responses without chain-of-thought reasoning:

```bash
cd scripts/validators
./validate_nocot.sh
```
More instructions about the validation pipeline can be found <a href="./scripts/validators/README.md">here</a> 
