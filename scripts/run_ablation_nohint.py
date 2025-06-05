import os
import argparse
from create_prompt_nohint import create_prompt_swebench, create_prompt_humaneval_cruxeval, create_prompt_classeval
from prompt import generator_lllm, generator_dsr1
from tqdm import tqdm
def main(model_key, category):
    root = f"../dataset/{category}"
    wr_root = f"../Results/experiment_results_output_nohint/{model_key.split('/')[-1]}/{category}"
    if not os.path.exists(wr_root):
        os.makedirs(wr_root)
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
        if dataset == 'swebench':
            code_path = os.path.join(root, d, 'main.py')
            input_path = os.path.join(root, d, 'input.txt')
            output_hint_path = os.path.join(root, d, 'output_hinted.txt')
            dependency_path = os.path.join(root, d, 'dependency.py')
            if os.path.exists(dependency_path):
                dependency = open(dependency_path, 'r').read()
            else:
                dependency = ""
            code = open(code_path, 'r').read()
            input_text = open(input_path, 'r').read()
            output_hint_text = open(output_hint_path, 'r').read()
            fun_name = d.split("@@")[-1]
            prompt = create_prompt_swebench(code, input_text, output_hint_text, fun_name, dependency)
        elif dataset in ["humaneval", "cruxeval"]:
            code_path = os.path.join(root, d, 'main.py')
            input_path = os.path.join(root, d, 'input.txt')
            code = open(code_path, 'r').read()
            input_text = open(input_path, 'r').read()
            prompt = create_prompt_humaneval_cruxeval(code, input_text)

        elif dataset == "avatar":
            code_path = os.path.join(root, d, 'main.py')
            input_path = os.path.join(root, d, 'input.txt')
            code = open(code_path, 'r').read()
            input_text = open(input_path, 'r').read()
            prompt = create_prompt_humaneval_cruxeval(code, input_text)
        elif dataset == "classeval":
            code_path = os.path.join(root, d, 'main.py')
            code = open(code_path, 'r').read()
            prompt = create_prompt_classeval(code)
        
        if model_key != "deepseek-r1":
            response = generator_lllm(model_key, prompt)
        else:
            response = generator_dsr1(prompt)
        
        
        response_full_path = os.path.join(wr_folder, 'response_full.txt')
        response_path = os.path.join(wr_folder, 'response.txt')
        if not os.path.exists(wr_folder):
            os.mkdir(wr_folder)
        with open(response_full_path, 'w') as wr1:
            wr1.write(str(prompt) + "\n\n" + str(response))
        with open(response_path, 'w') as wr2:
            wr2.write(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    parser.add_argument("--category", type=str, default='none')
    args = parser.parse_args()
    model_id = args.model
    category = args.category
    
    main(model_id, category)

            
    