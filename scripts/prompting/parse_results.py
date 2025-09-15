import os
import json
import argparse
from parsing_utils import extract_output_json_easy


def parse_results(model, task):
    model_id = model.split("/")[-1]
    root_response = f"../results/{task}/{model_id}"
    
    jsonl_path = f"../results/summary/{model_id}_{task}.jsonl"
    
    jsonl_writer = open(jsonl_path, 'w')
    
    for difficulty in ["difficult", "easy"]:
        folder = os.path.join(root_response, difficulty)
        for filename in os.listdir(folder):
            problem_index, _  = os.path.splitext(filename)
            file_path = os.path.join(folder, filename)
            response_text = open(file_path, 'r').read()
            parsed_json = {}
            try:
                if task == "output_prediction":
                    result = extract_output_json_easy(response_text, "output")
                else:
                    result = extract_output_json_easy(response_text, "input")
                if result["json"] is not None:
                    parsed_json = result['json']
                    # print(parsed_json)
                else:
                    print(f"Error in parsing {problem_index} in {model} on {difficulty}")
            except:
                print(f"Error in parsing {problem_index} in {model} on {difficulty}")
            result = {
                "model": model,
                "prediction": parsed_json,
                "problem_id": problem_index,
                "difficulty": difficulty
            }
            
            json_line = json.dumps(result)
            jsonl_writer.write(json_line + "\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    
    args = parser.parse_args()
    model = args.model
    task = args. task
    parse_results(model, task)
            