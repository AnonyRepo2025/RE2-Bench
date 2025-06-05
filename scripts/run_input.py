import os
import argparse
from create_prompt_input import create_prompt_swebench, create_prompt_humaneval_cruxeval, create_prompt_classeval, create_prompt_avatar, create_prompt_deepseekcoder
from prompt import generator_lllm, generator_dsr1, generator_vllm
from tqdm import tqdm
from vllm import LLM, SamplingParams
def main(model_key, category):
    root = f"../dataset/{category}"
    wr_root = f"../Results/experiment_results_input/{model_key.split('/')[-1]}/{category}"
    if not os.path.exists(wr_root):
        os.makedirs(wr_root)
    if model_key == "deepseek-ai/deepseek-coder-33b-instruct":
        cache_dir = "../"
        model = LLM(model=model_id, max_model_len=35000, download_dir=cache_dir, tensor_parallel_size=4)
    for d in tqdm(os.listdir(root)):
        wr_folder = os.path.join(wr_root, d)
        if os.path.exists(wr_folder):
            continue
        
        if d.startswith('Avatar'):
            dataset = 'avatar'
        elif d.startswith('ClassEval'):
            dataset = 'classeval'
        elif d.startswith('cruxeval'):
            dataset = 'cruxeval'
        elif d.startswith('HumanEval'):
            dataset = 'humaneval'
        else:
            dataset = 'swebench'
        try:
            if dataset == 'swebench':
                code_path = os.path.join(root, d, 'main.py')
                output_path = os.path.join(root, d, 'output.txt')
                input_hint_path = os.path.join(root, d, 'input_hinted.txt')
                dependency_path = os.path.join(root, d, 'dependency.py')
                if os.path.exists(dependency_path):
                    dependency = open(dependency_path, 'r').read()
                else:
                    dependency = ""
                code = open(code_path, 'r').read()
                output_text = open(output_path, 'r').read()
                input_hint_text = open(input_hint_path, 'r').read()
                fun_name = d.split("@@")[-1]
                prompt = create_prompt_swebench(code, output_text, input_hint_text, fun_name, dependency)
            elif dataset in ["humaneval", "cruxeval"]:
                code_path = os.path.join(root, d, 'main.py')
                output_path = os.path.join(root, d, 'output.txt')
                input_path = os.path.join(root, d, 'input.txt')
                code = open(code_path, 'r').read()
                output_text = open(output_path, 'r').read()
                input_text = open(input_path, 'r').read()
                fun_name = input_text.split("(")[0]
                prompt = create_prompt_humaneval_cruxeval(code, output_text, fun_name)
            elif dataset == "avatar":
                code_path = os.path.join(root, d, 'main.py')
                output_path = os.path.join(root, d, 'output.txt')
                code = open(code_path, 'r').read()
                output_text = open(output_path, 'r').read()
                prompt = create_prompt_avatar(code, output_text)
            elif dataset == "classeval":
                # continue
                code_path = os.path.join(root, d, 'main.py')
                output_path = os.path.join(root, d, 'output.txt')
                input_hint_path = os.path.join(root, d, 'input_hinted.txt')
                code = open(code_path, 'r').read()
                output_text = open(output_path, 'r').read()
                input_hint_text = open(input_hint_path, 'r').read()
                prompt = create_prompt_classeval(code, output_text, input_hint_text)

            if model_key == "deepseek-ai/deepseek-coder-33b-instruct":
                prompt = create_prompt_deepseekcoder(prompt)
                response = generator_vllm(model, prompt, 2000)
                
            elif model_key != "deepseek-r1":
                response = generator_lllm(model_key, prompt)
            else:
                response = generator_dsr1(prompt)
            try:
                with open(f"../Results/experiment_results_input/{model_key.split('/')[-1]}/{category}/{d}/prompt.txt",
                          "w", encoding="utf-8") as f:
                    f.write(prompt)
            except FileNotFoundError:
                pass
        except:
            response = "ERROR"    
            
        response_full_path = os.path.join(wr_folder, 'response_full.txt')
        response_path = os.path.join(wr_folder, 'response.txt')
        try:
            if not os.path.exists(wr_folder):
                os.mkdir(wr_folder)
            with open(response_full_path, 'w') as wr1:
                wr1.write(str(prompt) + "\n\n" + str(response))
            with open(response_path, 'w') as wr2:
                wr2.write(response)
        except:
            if not os.path.exists(wr_folder):
                os.mkdir(wr_folder)
            with open(response_full_path, 'w') as wr1:
                wr1.write("")
            with open(response_path, 'w') as wr2:
                wr2.write("")           
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    parser.add_argument("--category", type=str, default='none')
    args = parser.parse_args()
    model_id = args.model
    category = args.category
    main(model_id, category)