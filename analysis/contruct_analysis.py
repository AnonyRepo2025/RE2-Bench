import json
import os
import csv
from code_utils import read_dependency, read_code, find_io_index
from construct_extraction import *
from star_plots import star_plot

def prepare_code():
    wr_dir = "./code_dir"
    dataset_path = "../dataset/re2-bench/sampled_problems.json"
    dataset = json.load(open(dataset_path, 'r'))
    for key in dataset.keys():
        for d in dataset[key]:
            index = d 
            source = dataset[key][d]['benchmark']
            io_id = dataset[key][d]['input-output']
            
            source_code = read_code(index, source)
            if source == "Swebench":
                dependency_code = read_dependency(index)
                full_code = dependency_code + "\n\n" + source_code
            else:
                full_code = source_code
            
            code_path = f"{wr_dir}/code/{index}.py"
            code_full_path = f"{wr_dir}/code_full/{index}.py"
            with open(code_path, 'w') as f:
                f.write(source_code)
            with open(code_full_path, 'w') as f:
                f.write(full_code)

def get_constructs():
    overall_results = {}
    root = "./code_dir/code"
    for d in os.listdir(root):
        code_path = os.path.join(root, d)
        # print(code_path)
        index = code_path.split("/")[-1].removesuffix(".py")
        code = open(code_path, 'r').read()
        results = {
                "nested": 0,
                "if": 0,
                "for": 0,
                "while": 0,
                "try": 0,
                "switch": 0,
                "basic": 1,
                "nested_if": 0
            }
        try:
            if has_nested_loops(code):
                results["nested"] = 1
                results["basic"] = 0
            if has_unnested_if(code):
                results["if"] = 1
                results["basic"] = 0
            if has_for_loop(code):
                results["for"] = 1
                results["basic"] = 0
            if has_while_loop(code):
                results["while"] = 1
                results["basic"] = 0
            if has_try_except(code):
                results["try"] = 1
                results["basic"] = 0
            if has_nested_if(code):
                results["nested_if"] = 1
                results["basic"] = 0
        except Exception as e:
            print(code_path)
            print(e)
        overall_results[d.removesuffix(".py")] = results
    path = "./constructs.json"
    with open(path, 'w') as wr:
        json.dump(overall_results, wr, indent=4)




def load_input_data(model, id_map):
    results = {
        "difficult":{},
        "easy":{}
    }
    path = f"../results/validations/{model}_input.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    results["difficult"][case_name] = 1
                else:
                  results["easy"][case_name] = 1
            elif int(row["rs"]) == 0 and row["is_fn"] == "True":
                if row["difficulty_level"] == "difficult":
                    results["difficult"][case_name] = 1
                else:
                  results["easy"][case_name] = 1
            else:
                if row["difficulty_level"] == "difficult":
                    results["difficult"][case_name] = 0
                else:
                  results["easy"][case_name] = 0
    return results

def load_output_data(model, id_map):
    results = {
        "difficult": {},
        "easy": {}
    }
    path = f"../results/validations/{model}_output.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    results["difficult"][case_name] = 1
                else:
                  results["easy"][case_name] = 1
            else:
                if row["difficulty_level"] == "difficult":
                    results["difficult"][case_name] = 0
                else:
                  results["easy"][case_name] = 0
    return results





def add_label(construct_data, result_data):
    result = {}
    total_count = {
        'nested': 0, 'if': 0, 'for': 0, 'while': 0, 'try': 0, 'switch': 0, 'basic': 0, "nested_if": 0
    }
    correct_count = {
        'nested': 0, 'if': 0, 'for': 0, 'while': 0, 'try': 0, 'switch': 0, 'basic': 0, "nested_if": 0
    }
    for d in result_data.keys():
        
        if d not in construct_data:
            print(d)
            continue
        label = result_data[d]
        constructs = construct_data[d]
        for k in constructs.keys():
            if constructs[k] == 1:
                total_count[k] += 1
                if label == 1:
                    correct_count[k] += 1
    
    for k in total_count.keys():
        if total_count[k]>0:
            result[k] = correct_count[k] / total_count[k]
    return result
    
    

def construct_extractor(model, task, difficulty):
    id_map = id_mapping()
    constrcut_data = json.load(open("./constructs.json", 'r'))
    if task == "input_prediction":
        result_data = load_input_data(model, id_map)
    else:
        result_data = load_output_data(model, id_map)
    result_data = result_data[difficulty]

    result = add_label(constrcut_data, result_data)
    return result

def id_mapping():
    dataset_path = "../dataset/re2-bench/sampled_problems.json"
    dataset = json.load(open(dataset_path, 'r'))
    id_map = {}
    for key in dataset.keys():
        for d in dataset[key]:
            index = d 
            source = dataset[key][d]['benchmark']
            io_id = dataset[key][d]['input-output']
            target_id = find_io_index(index, source)
            id_map[target_id] = d
    return id_map


def collect_data(task, difficulty):
    results = {
        "o4-mini": construct_extractor("o4-mini", task, difficulty),
        "gpt-4.1": construct_extractor("gpt-4.1", task, difficulty),
        "gemini-2.5-pro": construct_extractor("gemini-2.5-pro", task, difficulty),
        "gemini-1.5-pro": construct_extractor("gemini-1.5-pro", task, difficulty),
        "deepseek-r1": construct_extractor("deepseek-reasoner", task, difficulty),
        "deepseek-coder": construct_extractor("deepseek-coder-33b-instruct", task, difficulty)
    }
    wr_path = f"./construct_{task}_{difficulty}.json"
    with open(wr_path, 'w') as wr:
        json.dump(results, wr, indent=4)


def plot(task, difficulty):
    json_path = f"./construct_{task}_{difficulty}.json"
    construct_data = json.load(open(json_path, 'r'))
    
    data_o4mini = construct_data["o4-mini"]
    data_gpt4 = construct_data["gpt-4.1"]
    data_gemini25 = construct_data["gemini-2.5-pro"]
    data_gemini15 = construct_data["gemini-1.5-pro"]
    data_deepseekr1 = construct_data["deepseek-r1"]
    data_deepseekcoder = construct_data["deepseek-coder"]
    
    labels = ["if", "for", "while", "try", "nested", "nested_if", "basic"]
    new_label_difficult = ["I", "F", "W", "T", "NL", "NI"]
    new_label_easy = ["I", "F", "W", "T", "NL", "NI", "B"]
    
    d_o4mini = [data_o4mini.get(label, 0) for label in labels if label in data_o4mini]
    d_gemini25 = [data_gemini25.get(label, 0) for label in labels if label in data_gemini25]
    d_deepseekr1 = [data_deepseekr1.get(label, 0) for label in labels if label in data_deepseekr1]
    d_gpt4 = [data_gpt4.get(label, 0) for label in labels if label in data_gpt4]
    d_gemini15 = [data_gemini15.get(label, 0) for label in labels if label in data_gemini15]
    d_deepseekcoder = [data_deepseekcoder.get(label, 0) for label in labels if label in data_deepseekcoder]
    
    if difficulty == "difficult":
        new_label = new_label_difficult
    else:
        new_label = new_label_easy
    
    data = [
        new_label,
        (
            task, [
                d_o4mini,
                d_gemini25,
                d_deepseekr1,
                d_gpt4,
                d_gemini15,
                d_deepseekcoder
            ]
        )
    ]
    labels = ('O4-mini','Gemini-2.5-Pro', 'DeepSeek-R1', 'GPT-4.1', 'Gemini-1.5-Pro', 'DeepSeek-Coder-Inst-33b')
    savepath = f"./construt_{task}_{difficulty}.jpeg"
    
    star_plot(data, len(new_label), labels, savepath)


