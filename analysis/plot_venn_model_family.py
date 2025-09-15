import csv 
import json 
from code_utils import find_io_index
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

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


def load_input_data(model, id_map):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_input.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(case_name)
                else:
                    easy_instances.append(case_name)
            if int(row["rs"]) == 0 and row["is_fn"] == "True":
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(case_name)
                else:
                    easy_instances.append(case_name)
    return difficulty_instances, easy_instances


def load_output_data(model, id_map):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_output.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = id_map[row['case_name']]
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(case_name)
                else:
                    easy_instances.append(case_name)
    return difficulty_instances, easy_instances



def plot_venn(list1, list2, title):
    plt.cla()
    set1 = set(list1)
    set2 = set(list2)

    venn = venn2([set1, set2], set_labels=('', ''))
    for text in venn.subset_labels:
        if text:  # avoid None
            text.set_fontsize(30)
    save_path = f"./model_family_{title}.jpg"
    plt.savefig(save_path,dpi=300, bbox_inches='tight', pad_inches=0)
    

def main(model_1, model_2):
    ## plot for difficult problems
    id_map = id_mapping()
    difficult_input_1, easy_input_1  = load_input_data(model_1, id_map)
    difficult_output_1, easy_output_1  = load_output_data(model_1, id_map)
    
    difficult_input_2, easy_input_2  = load_input_data(model_2, id_map)
    difficult_output_2, easy_output_2  = load_output_data(model_2, id_map)
    

    
    title1 = f"{model_1}_{model_2}_input_difficult"
    title2 = f"{model_1}_{model_2}_input_easy"
    title3 = f"{model_1}_{model_2}_output_difficult"
    title4 = f"{model_1}_{model_2}_output_easy"
    plot_venn(difficult_input_1, difficult_input_2, title1)
    plot_venn(easy_input_1, easy_input_2, title2)
    plot_venn(difficult_output_1, difficult_output_2, title3)
    plot_venn(easy_output_1, easy_output_2, title4)

main("deepseek-reasoner", "deepseek-coder-33b-instruct")
main("gemini-2.5-pro", "gemini-1.5-pro")
main("o4-mini", "gpt-4.1")