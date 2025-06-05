import json
import os
import matplotlib.pyplot as plt

def load_data(model_id):
    summary = {
        "high": [0, 0, 0],
        "medium":[0, 0, 0],
        "low": [0, 0, 0]
    }
    path_input_results = f"../Results/summary/input_{model_id}.json"
    path_output_results = f"../Results/summary/output_{model_id}.json"
    input_data = json.load(open(path_input_results, 'r'))
    output_data = json.load(open(path_output_results, "r"))
    
    high_ids = list(os.listdir("../dataset/high"))
    medium_ids = list(os.listdir("../dataset/medium"))
    low_ids = list(os.listdir("../dataset/low"))
    
    correct_input, correct_output = [], []
    for k in input_data:
        if input_data[k] == 1:
            correct_input.append(k)
    for j in output_data:
        if output_data[j] == 1:
            correct_output.append(j)
    
    unique_input = set(correct_input) - set(correct_output)
    unique_output = set(correct_output) - set(correct_input)
    overlapped = set(correct_input) & set(correct_output)
    
    for i in unique_input:
        if i in high_ids:
            summary["high"][0] += 1
        elif i in medium_ids:
            summary["medium"][0] += 1
        else:
            summary["low"][0] += 1
    
    for i in unique_output:
        if i in high_ids:
            summary["high"][1] += 1
        elif i in medium_ids:
            summary["medium"][1] += 1
        else:
            summary["low"][1] += 1
    
    for i in overlapped:
        if i in high_ids:
            summary["high"][2] += 1
        elif i in medium_ids:
            summary["medium"][2] += 1
        else:
            summary["low"][2] += 1
    
    return summary     

def donuts(results, label):
    plt.cla()
    def make_labels(data):
        return [str(v) if v > 0 else '' for v in data]
    # Data for each ring
    data1 = results["high"]   # Inner ring
    data2 = results["medium"]   # Middle ring
    data3 = results["low"]  # Outer ring

    # Colors (optional, for visual clarity)
    colors =  ['plum', "limegreen", "navajowhite"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.4

    # First (inner) ring
    ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.52, textprops={'fontsize': 40})

    # Second (middle) ring
    if sum(data2) > 0:
        ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.70, textprops={'fontsize': 40})

    # Third (outer) ring
    ax.pie(data3, radius=1+width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data3), labeldistance=0.80, textprops={'fontsize': 40})


    # ax.text(0, 0, 'High', ha='center', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 0.8, 'Medium', ha='left', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 1.2, 'Low', ha='center', va='center', fontsize=16, fontweight='bold')

    save_path = f"../analysis/figs/donut-overlapped-{label}.jpg"
    plt.savefig(save_path,dpi=500)

def main(model_id):
    results = load_data(model_id)
    donuts(results, model_id)

main("o4-mini-2025-04-16")
main("gemini-2.5-pro-preview-03-25")
main("deepseek-r1")
main("gpt-4.1-2025-04-14")
main("gemini-1.5-pro")
main("deepseek-coder-33b-instruct")