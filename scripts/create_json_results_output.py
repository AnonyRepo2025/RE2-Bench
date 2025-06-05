import json
import os
import argparse

def parse_output(model_id):
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
        root_dir = f"../Results/experiment_results_output/{model_id.split('/')[-1]}/{cat}"
        for d in os.listdir(root_dir):
            response_path = os.path.join(root_dir, d, 'response.txt')
            if not os.path.exists(response_path):
                summary[d] = 0
                continue
            response_text = open(response_path, 'r').read()

            if "[ANSWER]" in response_text and "[/ANSWER]" in response_text or ("[OUTPUT]" in response_text and "[/OUTPUT]" in response_text):
                if model_id in ["o4-mini-2025-04-16", "gemini-1.5-pro", "gemini-2.5-pro-preview-03-25", "deepseek-r1", "deepseek-coder-33b-instruct"]:
                    if "[ANSWER]" in response_text and "[/ANSWER]" in response_text:
                        json_str = response_text.split("[ANSWER]")[1].split("[/ANSWER]")[0].replace("\n", "")
                    if "[OUTPUT]" in response_text and "[/OUTPUT]" in response_text:
                        json_str = response_text.split("[OUTPUT]")[1].split("[/OUTPUT]")[0].replace("\n", "")                    
                    json_str = json_str.strip("```json").strip("```")
                    try:
                        output_data = json.loads(json_str)
                    except:
                        output_data = {"output":""}
                        summary[d] = 0
                        print(response_path)
                if model_id == "gpt-4.1-2025-04-14":
                    if "[ANSWER]" in response_text and "[/ANSWER]" in response_text:
                        json_str = response_text.split("[ANSWER]")[1].split("[/ANSWER]")[0].replace("\n", "")
                    if "[OUTPUT]" in response_text and "[/OUTPUT]" in response_text:
                        json_str = response_text.split("[OUTPUT]")[1].split("[/OUTPUT]")[0].replace("\n", "")  
                    if json_str.startswith("```"):
                        json_str = json_str.strip("```")
                    try:
                        output_data = json.loads(json_str)
                    except:
                        try:
                            json_str = json_str.replace("True", "true").replace("None", "null").replace("False", "false")
                            output_data = json.loads(json_str)
                        except:
                            output_data = {"output":""}
                            print(response_path)
            else:
                output_data = {"output":""}
                print(response_path)
            wr_path = os.path.join(root_dir, d, 'response.json')
            with open(wr_path, 'w') as wr:
                json.dump(output_data, wr, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    args = parser.parse_args()
    model_id = args.model
    model_id = model_id.split('/')[-1]
    parse_output(model_id)