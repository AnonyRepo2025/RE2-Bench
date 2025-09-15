import csv
import json 
from code_utils import find_io_index
import matplotlib.pyplot as plt

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

def load_complexity(construct_data):
    results = {}
    path = "../scripts/categorization/all_benchmarks.csv"
    data = read_csv(path)
    for d in data: 
        pid, _ = clean_indices(d[0])
        attributes = d[1:10]
        
        if pid not in construct_data:
            continue 
        results[pid] = []
        if construct_data[pid]["if"] > 0 or construct_data[pid]["nested_if"] > 0:
            results[pid].append("conditional")
        if construct_data[pid]["for"] > 0 or construct_data[pid]["nested"] > 0 or construct_data[pid]["while"]>0:
            results[pid].append("recursion")
        if int(attributes[4]) > 0 or int(attributes[5]) > 0 or int(attributes[6]) > 0:
            results[pid].append("call")
        if int(attributes[3]) > 0:
            results[pid].append("structural")
        if int(attributes[7]) > 0 or int(attributes[8]) > 0:
            results[pid].append("types")        
    return results

def load_construct_data():
    json_path = "./constructs.json"
    construct_data = json.load(open(json_path, "r"))
    return construct_data



def false_prediction_analysis(model, task):
    counter = {
        "model": model,
        "task": task,
        "conditional": 0,
        "recursion": 0,
        "call": 0,
        "structural": 0,
        "types": 0
    }
    false_reports = {}
    id_map = id_mapping()
    construct_data = load_construct_data()
    causes = load_complexity(construct_data)

    if task == "input_prediction":
        prediction_result = load_input_data(model, id_map)
    else:
        prediction_result = load_output_data(model, id_map)
    false_predictions = [k for k,v in prediction_result.items() if v == 0]
    
    for k in false_predictions:
        reasons = causes.get(k, [])
        false_reports[k] = reasons
        for r in reasons:
            counter[r] += 1
    
    report_path = f"./false_prediction_reports/{model}_{task}_false_reports.json"
    with open(report_path, 'w') as wr:
        json.dump(false_reports, wr, indent=4)
    return counter


def collect_false_predictions():
    data = []
    models = ["o4-mini", "gemini-2.5-pro", "deepseek-reasoner", "gpt-4.1", "gemini-1.5-pro", "deepseek-coder-33b-instruct"]
    for model in models:
        result_input = false_prediction_analysis(model, "input_prediction")
        result_output = false_prediction_analysis(model, "output_prediction")
        data.append(result_input)
        data.append(result_output)
    fieldnames = ["model", "task", "conditional", "recursion", "call", "structural", "types"]
    csv_file = "./false_prediction_analysis.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def donuts_two_rings(results, save_path):
    plt.cla()

    def make_labels(data):
        return [str(v) if v > 3 else '' for v in data]

    data1 = results["input_prediction"]   # Inner ring
    data2 = results["output_prediction"]   # Middle ring

    colors =  ['#b4a7d6', '#9fc5e8', '#b7d7a9', '#ffe599', "#f9cb9c"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    width = 0.4

    # labeldistance that centers text in each ring
    ld_inner = 1 - width / (2 * (1 - width))   # = 2/3 for width=0.4
    ld_outer = 1 - width / 2                   # = 0.8 for width=0.4

    # Inner ring
    wedges, texts = ax.pie(
        data1,
        radius=1 - width,
        colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1),
        labeldistance=ld_inner,
        textprops={'fontsize': 30, 'ha': 'center', 'va': 'center'},
    )

    # Outer ring
    if sum(data2) > 0:
        wedges2, texts2 = ax.pie(
            data2,
            radius=1,
            colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2),
            labeldistance=ld_outer,
            textprops={'fontsize': 30, 'ha': 'center', 'va': 'center'},
        )
    

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)


def plot_per_llm():
    results = {}
    with open('./false_prediction_analysis.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            task = row["task"]
            model = row["model"]
            counts = [int(row["conditional"]), int(row["recursion"]), int(row["structural"]), int(row["call"]), int(row["types"])]
            
            if model not in results:
                results[model] = {}
            results[model][task] = counts
    
    for k in results:
        input_failures = results[k]['input_prediction']
        output_failures = results[k]['output_prediction']
        
        data = {
            "input_prediction": input_failures,
            "output_prediction": output_failures
        }
        
        save_path= f"./{k}_failure_analysis.jpeg"
        
        donuts_two_rings(data, save_path)


if __name__ == "__main__":
    collect_false_predictions()