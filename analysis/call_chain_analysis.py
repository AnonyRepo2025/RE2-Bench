import ast
import os
import json
import matplotlib.pyplot as plt
import statistics
def count_functions_in_file(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read(), filename=filename) 
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    return max(1,len(function_defs)) ## for Avatar code

def collect_dependencies():
    results = {}
    root = "../dataset"
    categories = ["high", "low", "medium"]
    for cat in categories:
        folder = os.path.join(root, cat)
        for d in os.listdir(folder):
            main_path = os.path.join(folder, d, 'main.py')
            dep_path = os.path.join(folder, d, 'dependency.py')
            dep_counter = 0
            dep_counter += count_functions_in_file(main_path)
            if os.path.exists(dep_path):
                dep_counter += count_functions_in_file(dep_path)
            results[d] = dep_counter
    save_path = "../Results/summary/dep.json"
    with open(save_path, 'w') as wr:
        json.dump(results, wr, indent=4)
    
def load_data(model_id):
    succ_deps, fail_deps = [], []
    result_path = f"../Results/summary/input_{model_id}.json"
    dep_path = "../Results/summary/dep.json"
    dep_data = json.load(open(dep_path, 'r'))
    results = json.load(open(result_path, 'r'))
    for k in results:
        if results[k]:
            succ_deps.append(dep_data[k])
        else:
            dep = dep_data[k]
            fail_deps.append(dep)
    return succ_deps, fail_deps
    # data = [succ_deps, fail_deps]
    # plt.figure(figsize=(2,4))
    # plt.boxplot(data)
    # savepath = f"/home/changshu/RE2-Bench/analysis/figs/call_chain_{model_id}.jpg"
    # plt.yticks([])
    # plt.xticks([1, 2], ['succ', 'fail'])
    # plt.xticks(fontsize=20)
    # plt.savefig(savepath, dpi=500)


def plot_box():
    succ_o4, fail_o4 = load_data("o4-mini-2025-04-16")
    succ_g2, fail_g2 = load_data("gemini-2.5-pro-preview-03-25")
    succ_dsr, fail_dsr = load_data("deepseek-r1")
    succ_gpt, fail_gpt = load_data("gpt-4.1-2025-04-14")
    succ_g1, fail_g1 = load_data("gemini-1.5-pro")
    succ_dsc, fail_dsc = load_data("deepseek-coder-33b-instruct")
    
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
    savepath = f"../analysis/figs/call_chain_input.jpg"
    # plt.yticks([])
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail', 'succ', 'fail'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=500)
    
plot_box()

    