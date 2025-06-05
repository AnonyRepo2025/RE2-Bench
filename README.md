## Installation
Run the following commands to create the environment:
```
conda create -n re2-bench python=3.9
pip install -r requirements.txt
conda activate re2-bench
```

Add your `OpenAI Key`, `Gemini Key`, and `DeepSeek Key` to `.env`.
```
OPENAIKEY=<YOUR OPENAI KEY>
GEMINIKEY=<YOUR GEMINI KEY>
DEEPSEEKKEY=<YOUR DEEPSEEKKEY>
```

We provide a Dockerfile to reproduce the results of RE2-bench.
Execute the following to create a docker image and execute the container in interactive mode:
```
docker build -t re2bench .
docker run -it re2bench bash
```

If you are using MacOS with an Apple chip, please consider adding `--platform=linux/amd64` in docker build.

## Reproducing Results
### Main Results(Table.3)
#### Output prediction
```
cd scripts
bash run_output_prediction.sh <MODEL_ID>
```
Choose a model from `{o4-mini-2025-04-16, gpt-4.1-2025-04-14, gemini/gemini-2.5-pro-preview-03-25, gemini/gemini-1.5-pro, deepseek-r1, deepseek-ai/deepseek-coder-33b-instruct}`

#### Input prediction
```
cd scripts
bash run_input_prediction.sh <MODEL_ID>
```




### Ablation Study

#### Prompt without CoT(Table.5)
```
cd scripts
bash run_ablation_wocot.sh <MODEL_ID>
```

#### Prompt without Input/Output Structure(Table.5)
```
cd scripts
bash run_ablation_wohint.sh <MODEL_ID>
```
