import csv 
import json
from code_utils import find_io_index
from scipy.stats import pointbiserialr
from scipy.stats import spearmanr
def read_csv(csv_path, with_header=False):
    '''read csv file'''
    data = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        if with_header:
            header = next(reader)
        # print(header)
        for row in reader:
            data.append(row)
    return data

def clean_indices(index_string):
    if "swebench" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "Swebench"
    elif "classeval" in index_string:
        return index_string.split("/")[-2], "Classeval"
    elif "cruxeval" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "cruxeval"
    elif "HumanEval" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "HumanEval"
    elif "Avatar" in index_string:
        return index_string.split("/")[-2], "Avatar"

def load_complexity():
    results = {}
    path = "../scripts/categorization/all_benchmarks.csv"
    data = read_csv(path)
    for d in data: 
        pid, _ = clean_indices(d[0])
        results[pid] = d[1:10]
    return results

def load_input_data(model, id_map):
    results = {}
    path = f"../results/validations/{model}_input.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                results[case_name] = 1
            elif int(row["rs"]) == 0 and row["is_fn"] == "True":
                results[case_name] = 1
            else:
                results[case_name] = 0
    return results

def load_output_data(model, id_map):
    results = {}
    path = f"../results/validations/{model}_output.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                results[case_name] = 1
            else:
                results[case_name] = 0
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

def load_data(model, task):
    id_map = id_mapping()
    complexity_data = load_complexity()
    if task == "input_prediction":
        results = load_input_data(model, id_map)
    else:
        results = load_output_data(model, id_map)
    summary = {"c1": {}, "c2": {}, "c3": {}, "c4": {}, "c5": {}, "c6": {}, "c7": {}, "c8": {}, "c9": {}}
    for r in results:
        if r in complexity_data:
            for i in range(len(complexity_data[r])):
                key = f"c{i+1}"
                value = complexity_data[r][i]
                
                if "complexity" not in summary[key]:
                    summary[key]["complexity"] = []
                    summary[key]["results"] = []
                summary[key]["results"].append(results[r])
                summary[key]["complexity"].append(int(value))
    return summary


def compute_corr(model_id, label, task):
    data = load_data(model_id, task)[label]
    corr, p_value = spearmanr(data["results"], data["complexity"])
    return round(float(corr),2)

if __name__ == "__main__":
    model = "gpt-4.1"
    task = "input_prediction"
    
    results_dict = {
        "o4-mini":[],
        "gemini-2.5-pro": [],
        "deepseek-reasoner": [],
        "gemini-1.5-pro": [],
        "gpt-4.1": [],
        "deepseek-coder-33b-instruct": []
    }
    
    for i in range(1, 10):
        label = f"c{i}"
        for k in results_dict.keys():
            corr = compute_corr(k, label, task)
            results_dict[k].append(corr)
    cor_path = f"./correlation_{task}.json"
    if not os.apth.exists(cor_path):
        with open(cor_path, 'w') as wr:
            json.dump(results_dict, wr, indent=4)
