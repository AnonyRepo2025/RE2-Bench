import matplotlib.pyplot as plt
import csv

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

def donuts(results, label):
    plt.cla()
    def make_labels(data):
        return [str(v) if v > 0 else '' for v in data]
    # Data for each ring
    data1 = results["strong"]   # Inner ring
    data2 = results["medium"]   # Middle ring
    data3 = results["weak"]  # Outer ring

    # Colors (optional, for visual clarity)
    colors =  ['salmon', 'orange', 'gold', 'lightgreen', "skyblue"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.4

    # First (inner) ring
    ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.52, textprops={'fontsize': 32})

    # Second (middle) ring
    if sum(data2) > 0:
        ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.70, textprops={'fontsize': 32})

    # Third (outer) ring
    ax.pie(data3, radius=1+width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data3), labeldistance=0.80, textprops={'fontsize': 32})


    # ax.text(0, 0, 'High', ha='center', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 0.8, 'Medium', ha='left', va='center', fontsize=16, fontweight='bold')
    # ax.text(0, 1.2, 'Low', ha='center', va='center', fontsize=16, fontweight='bold')

    save_path = f"../analysis/figs/donut-{label}.jpg"
    plt.savefig(save_path,dpi=500)
    

def collect_donut_data():
    start, end=1, 10
    summary = {}
    '''
    {
        "$dataset": {
            "$property": {
                "strong": N, "medium": N, "low": N
            }
        }
    }
    '''
    tagged_data_path = "../Results/all_benchmarks-tagged.csv"
    tagged_data = read_csv(tagged_data_path)
    for data in tagged_data:
        if 'Avatar' in data[0]:
            dataset = 'avatar'
        elif 'classeval' in data[0]:
            dataset = 'classeval'
        elif 'cruxeval' in data[0]:
            dataset = 'cruxeval'
        elif 'HumanEval' in data[0]:
            dataset = 'humaneval'
        elif 'swebench' in data[0]:
            dataset = 'swebench'
        if dataset not in summary:
            summary[dataset] = {}
        for i in range(start, end):
            label = f"c{i}"
            info = data[i]
            if label not in summary[dataset]:
                summary[dataset][label] = {
                    "strong":0,
                    "medium":0,
                    "weak": 0
                }
            summary[dataset][label][info] += 1 
    return summary
            
def plot_donut(data, label):
    results = {
        "dataset":[],
        "strong":[],
        "medium":[],
        "weak": []
    }
    for dataset in data.keys():
        results["dataset"].append(dataset)
        results["strong"].append(data[dataset][label]["strong"])
        results["medium"].append(data[dataset][label]["medium"])
        results["weak"].append(data[dataset][label]["weak"])
    # print(results["weak"][-2], sum(results["weak"]))
    donuts(results, label)
