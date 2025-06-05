import json
import os 
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

def load_data(model_id, constraints):
    results = []
    path_input_results = f"../Results/summary/input_{model_id}.json"
    path_output_results = f"../Results/summary/output_{model_id}.json"
    
    results_input = json.load(open(path_input_results, 'r'))
    results_output = json.load(open(path_output_results, 'r'))
    
    for i in results_input:
        if results_input[i] == 1 and i in constraints:
            results.append(i+"_input")
    for j in results_output:
        if results_output[j] == 1 and j in constraints:
            results.append(j+"_output")
    return results

def plot_venn(list1, list2, title):
    plt.cla()
    set1 = set(list1)
    set2 = set(list2)

    venn = venn2([set1, set2], set_labels=('', ''))
    for text in venn.subset_labels:
        if text:  # avoid None
            text.set_fontsize(30)
    save_path = f"../analysis/figs/{title}.jpg"
    plt.savefig(save_path,dpi=300, bbox_inches='tight', pad_inches=0)
    
def main(model_id1, model_id2):
    high_ids = list(os.listdir("../dataset/high"))
    medium_ids = list(os.listdir("../dataset/medium"))
    low_ids = list(os.listdir("../dataset/low"))
    
    results_1 = load_data(model_id1, high_ids)
    results_2 = load_data(model_id2, high_ids)
    title = f"{model_id1}_{model_id2}_high"
    plot_venn(results_1, results_2, title)
    
    results_1 = load_data(model_id1, medium_ids)
    results_2 = load_data(model_id2, medium_ids)
    title = f"{model_id1}_{model_id2}_medium"
    plot_venn(results_1, results_2, title)
    
    results_1 = load_data(model_id1, low_ids)
    results_2 = load_data(model_id2, low_ids)
    title = f"{model_id1}_{model_id2}_low"
    plot_venn(results_1, results_2, title)
    
main("deepseek-r1", "deepseek-coder-33b-instruct")