import json
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 2.0

meta_data_dataset = {
    'high': {
        'swebench': 36,
        'avatar': 3,
        'cruxeval': 0,
        'humaneval': 0,
        'classeval': 11
    },
    'medium': {
        'swebench': 13,
        'avatar': 4,
        'cruxeval': 12,
        'humaneval': 10,
        'classeval': 11
    },
    'low': {
        'swebench': 10,
        'avatar': 10,
        'cruxeval': 10,
        'humaneval': 10,
        'classeval': 10
    },    
}
def count_dataset(items, cat):
    count_swebench = items.count("swebench")
    count_classeval = items.count("classeval")
    count_avatar = items.count("avatar")
    count_cruxeval = items.count("cruxeval")
    count_humaneval = items.count("humaneval")
    return [count_swebench, meta_data_dataset[cat]["swebench"]-count_swebench,
        count_classeval, meta_data_dataset[cat]["classeval"]- count_classeval,
        count_avatar, meta_data_dataset[cat]["avatar"] - count_avatar,
        count_cruxeval, meta_data_dataset[cat]["cruxeval"] - count_cruxeval,
        count_humaneval, meta_data_dataset[cat]["humaneval"] - count_humaneval]

def load_model(model_id):
    data_path = f"../Results/summary/output_dataset_{model_id}.json"
    json_data = json.load(open(data_path, 'r'))
    data_high = json_data["high"]
    data_medium = json_data["medium"]
    data_low = json_data["low"]
    
    count_high = count_dataset(data_high, "high")
    count_low = count_dataset(data_low, "low")
    count_medium = count_dataset(data_medium, "medium")
    
    results = {
        "strong": count_high,
        "medium": count_medium,
        "weak": count_low
    }
    # print(results)
    donuts(results, model_id)
    

def donuts(results, label):
    plt.cla()
    def make_labels(data):
        return [str(v) if v >= 3 else '' for v in data]
    # Data for each ring
    data1 = results["strong"]   # Inner ring
    data2 = results["medium"]   # Middle ring
    data3 = results["weak"]  # Outer ring

    # Colors (optional, for visual clarity)
    colors = ['salmon', 'salmon','orange', 'orange', 'gold', 'gold', 'lightgreen', 'lightgreen' ,"skyblue", 'skyblue']
    hatches = ["", "O", "", "O","", "O","", "O", "", "O"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.4

    # First (inner) ring
    patches = ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.55, textprops={'fontsize': 40})
    for i in range(len(patches[0])):
        patches[0][i].set(hatch = hatches[i])
    # Second (middle) ring
    if sum(data2) > 0:
        patches = ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.76, textprops={'fontsize': 40})
        for i in range(len(patches[0])):
            patches[0][i].set(hatch = hatches[i])

    # Third (outer) ring
    patches = ax.pie(data3, radius=1+width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data3), labeldistance=0.80, textprops={'fontsize': 40})
    for i in range(len(patches[0])):
        patches[0][i].set(hatch = hatches[i])


    # ax.text(0, 0, 'High', ha='center', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 0.8, 'Medium', ha='left', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 1.2, 'Low', ha='center', va='center', fontsize=16, fontweight='bold')

    save_path = f"../analysis/figs/success-{label}-output.jpg"
    plt.savefig(save_path,dpi=300)
    

# load_model("gemini-1.5-pro")
# load_model("gpt-4.1-2025-04-14")
# load_model("gemini-2.5-pro-preview-03-25")
load_model("o4-mini-2025-04-16")
# load_model("deepseek-r1")
# load_model("deepseek-coder-33b-instruct")
