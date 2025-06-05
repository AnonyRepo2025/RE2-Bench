import json
import os
import argparse

def parse_input(model_id):
    categories = ['high', 'medium', 'low']
    summary = {}
    success_count = {
        'high':0,
        'medium': 0,
        'low': 0
    }
    success_dict = {
        'high': [],
        'medium': [],
        'low': []        
    }
    for cat in categories:
        root_dir = f"../Results/experiment_results_nocot_input/{model_id.split('/')[-1]}/{cat}"
        for d in os.listdir(root_dir):
            response_path = os.path.join(root_dir, d, 'response.txt')
            if not os.path.exists(response_path):
                summary[d] = 0
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
                        ground_truth_value = ground_truth_value.replace("\n", "").replace('"', "'").replace(" ","")
                        response_str = response_str.replace("\n","").replace('"', "'").replace(" ","")
                        if str(response_str) == ground_truth_value:
                            success_count[cat] += 1
                            success_dict[cat].append(dataset)
                            summary[d] = 1
                        else:
                            summary[d] = 0
                            if cat == "medium":
                                if d == "cruxeval@sample_107":
                                    print(response_str)
                                    print(ground_truth_value)
                    except:
                        summary[d] = 0
                        pass
                       
                else:
                    ground_truth_path = f"../dataset/{cat}/{d}/input.txt"
                    ground_truth_value = json.loads(open(ground_truth_path, 'r').read())
                    try:
                        input_data = json.loads(response_str)
                        if input_data == ground_truth_value:
                            
                            success_count[cat] += 1
                            success_dict[cat].append(dataset)
                            summary[d] = 1
                        else:
                            summary[d] = 0
                            
                    except:
                        summary[d] = 0
            else:
                summary[d] = 0
    print(success_count)      
    print((success_count["high"] + success_count["medium"] + success_count["low"])/150)
    
    for k in success_dict:
        print(k)
        print("swebench:", success_dict[k].count("swebench"))
        print("classeval:", success_dict[k].count("classeval"))
        print("avatar:", success_dict[k].count("avatar"))
        print("crxueval:", success_dict[k].count("cruxeval"))
        print("humaneval:", success_dict[k].count("humaneval"))
        
    summary_path = f"../Results/summary/input_{model_id}_nocot.json"
    with open(summary_path, 'w') as wr:
        json.dump(summary, wr, indent=4)

    # dataset_summary_path = f"../Results/summary/input_dataset_{model_id}.json"
    # with open(dataset_summary_path, 'w') as wr1:
    #     json.dump(success_dict, wr1, indent=4)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    args = parser.parse_args()
    model_id = args.model
    model_id = model_id.split('/')[-1]
    parse_input(model_id)