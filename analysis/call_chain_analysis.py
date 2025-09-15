import ast 
import os
import json
import csv
from code_utils import find_io_index
import matplotlib.pyplot as plt
def count_functions_in_file(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read(), filename=filename) 
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    return max(1,len(function_defs)) ## for Avatar code

def collect_dependencies():
    results = {}
    folder = "./code_dir/code_full"
    for d in os.listdir(folder):
        main_path = os.path.join(folder, d)
        dep_counter =  count_functions_in_file(main_path)
        results[d.removesuffix(".py")] = dep_counter
    path = "./dependencies.json"
    with open(path, 'w') as wr:
        json.dump(results, wr, indent=4)

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

def call_chain_counter(model, task):
    succ_deps, fail_deps = [], []
    id_map = id_mapping()
    call_data = json.load(open("./dependencies.json", 'r'))
    if task == "input_prediction":
        result_data = load_input_data(model, id_map)
    else:
        result_data = load_output_data(model, id_map)
    for k in result_data:
        for case in result_data[k]:
            if case not in call_data:
                continue
            if result_data[k][case] == 1:
                succ_deps.append(call_data[case])
            else:
                fail_deps.append(call_data[case])
    return succ_deps, fail_deps

def plot_box(task):
    succ_o4, fail_o4 = call_chain_counter("o4-mini", task)
    succ_g2, fail_g2 = call_chain_counter("gemini-2.5-pro", task)
    succ_dsr, fail_dsr = call_chain_counter("deepseek-reasoner", task)
    succ_gpt, fail_gpt = call_chain_counter("gpt-4.1", task)
    succ_g1, fail_g1 = call_chain_counter("gemini-1.5-pro",task)
    succ_dsc, fail_dsc = call_chain_counter("deepseek-coder-33b-instruct", task)
    
    data = [succ_dsc, fail_dsc, succ_g2, fail_g2, succ_dsr, fail_dsr, succ_gpt, fail_gpt, succ_g1, fail_g1,  succ_o4, fail_o4]
    
    plt.figure(figsize=(6, 2))
    bp = plt.boxplot(
        data,
        showmeans=True,
        widths=0.7,
        patch_artist=True,
        meanline=True
        )
    colors = ['lightyellow', 'lightyellow', 'greenyellow', "greenyellow", "aquamarine", "aquamarine", "lightpink", "lightpink", "mistyrose", "mistyrose", "cyan", "cyan"]
    savepath = f"./call_chain_{task}.jpg"
    # plt.yticks([])
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=500)
if __name__ == "__main__":
    # collect_dependencies()
    plot_box("input_prediction")
    plot_box("output_prediction")