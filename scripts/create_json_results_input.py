import json
import os
import argparse
import ast

def object_hook(obj):
    return str(obj)

def parse_input(model_id):
    categories = ['high', 'medium', 'low']
    for cat in categories:
        root_dir = f"../Results/experiment_results_input/{model_id.split('/')[-1]}/{cat}"
        for d in os.listdir(root_dir):
            response_path = os.path.join(root_dir, d, 'response.txt')
            if not os.path.exists(response_path):
                continue
            response_text = open(response_path, 'r').read()
            if "[ANSWER]" in response_text and "[/ANSWER]" in response_text:
                if model_id in ["o4-mini-2025-04-16", "gemini-1.5-pro", "gemini-2.5-pro-preview-03-25", "deepseek-r1", "gpt-4.1-2025-04-14", "deepseek-coder-33b-instruct"]:
                    response_str = response_text.split("[ANSWER]")[1].split("[/ANSWER]")[0].replace("\n", "")
                    if "[INPUT]" in response_str:
                        response_str = response_str.split("[INPUT]")[1].split('[/INPUT]')[0]
                    if response_str.startswith("```python"):
                        response_str = response_str.split("```python")[1].split('```')[0]
                    if response_str.startswith("```json"):
                        response_str = response_str.split("```json")[1].split('```')[0]
                    if response_str.startswith("```"):
                        response_str = response_str.split("```")[1].split('```')[0]       

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
                
                if dataset != "swebench":
                    ground_truth_path = f"../dataset/{cat}/{d}/input.txt"
                    ground_truth_value = open(ground_truth_path, 'r').read()
                    try:
                        response_str = response_str.replace("\n","").replace('"', "'").replace(" ","")
                        # print(response_str)
                        input_data = {"input": response_str}
                    except:
                        print(response_path)
                        input_data = {"input": ""}
                        pass
                        # print(response_path)
                       
                else:
                    try:
                        input_data = json.loads(response_str)
                        # print(input_data)
                    except Exception as e:
                        print(response_path)
                        input_data = {"input": ""}
            else:
                input_data = {"input": ""}
            wr_path = os.path.join(root_dir, d, 'response.json')
            
            with open(wr_path, 'w') as wr:
                json.dump(input_data, wr, indent=4)
    # print(success_count)      
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    args = parser.parse_args()
    model_id = args.model
    model_id = model_id.split('/')[-1]
    parse_input(model_id)
