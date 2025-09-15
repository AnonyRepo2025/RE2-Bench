import csv
import json
from utils import  donuts_overlap
meta_data_dataset = {
    'high': {
        'Swebench': 129,
        'Avatar': 6,
        'cruxeval': 0,
        'HumanEval': 0,
        'Classeval': 20
    },
    'low': {
        'Swebench': 8,
        'Avatar': 8,
        'cruxeval': 104,
        'HumanEval': 8,
        'Classeval': 27
    },    
}


def count_dataset(items, cat):
    count_swebench = items.count("Swebench")
    count_classeval = items.count("Classeval")
    count_avatar = items.count("Avatar")
    count_cruxeval = items.count("cruxeval")
    count_humaneval = items.count("HumanEval")
    return [count_swebench, meta_data_dataset[cat]["Swebench"]-count_swebench,
        count_classeval, meta_data_dataset[cat]["Classeval"]- count_classeval,
        count_avatar, meta_data_dataset[cat]["Avatar"] - count_avatar,
        count_cruxeval, meta_data_dataset[cat]["cruxeval"] - count_cruxeval,
        count_humaneval, meta_data_dataset[cat]["HumanEval"] - count_humaneval]

def load_input_data(model):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_input.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["case_name"])
                else:
                    easy_instances.append(row["case_name"])
            if int(row["rs"]) == 0 and row["is_fn"] == "True":
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["case_name"])
                else:
                    easy_instances.append(row["case_name"])
    return difficulty_instances, easy_instances

def load_output_data(model):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_output.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["case_name"])
                else:
                    easy_instances.append(row["case_name"])
    return difficulty_instances, easy_instances

def find_overlep(ids_input, ids_output):
    unique_input = set(ids_input) - set(ids_output)
    unique_output = set(ids_output) - set(ids_input)
    overlapped = set(ids_input) & set(ids_output)
    return [len(unique_input), len(unique_output), len(overlapped)], unique_output

if __name__ == "__main__":
    # models = ["gpt-4-turbo", "gemini/gemini-1.5-pro", "gemini/gemini-2.5-pro", "deepseek/deepseek-reasoner"]
    models = ["gpt-4.1", "gemini-1.5-pro", "gemini-2.5-pro", "deepseek-reasoner", "o4-mini", "deepseek-coder-33b-instruct"]
    for model in models:
        difficult_instances_input, easy_instances_input = load_input_data(model)
        difficult_instances_output, easy_instances_output = load_output_data(model)
        
        difficult_data, unique_output_difficult = find_overlep(difficult_instances_input, difficult_instances_output)
        easy_data, unique_output_easy = find_overlep(easy_instances_input, easy_instances_output)
        
        save_path = f"./figs/overlap/{model}.jpeg"
        results = {
            "high": difficult_data,
            "low": easy_data,
        }
        donuts_overlap(results, save_path)
        
        unique_output_swe = []
        for i in list(unique_output_difficult) + list(unique_output_easy):
            if "@@" in i:
                unique_output_swe.append(i.removesuffix(".jsonl"))
        with open(f'./unique_output_swe/{model}.json', 'w') as wr:
            json.dump(unique_output_swe, wr, indent=4)
       