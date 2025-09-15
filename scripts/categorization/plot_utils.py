import matplotlib.pyplot as plt

def donuts(results, label, save_path):
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

    plt.savefig(save_path, dpi=500)
    
    
def donuts_two_rings(results, label, save_path):
    plt.cla()
    def make_labels(data):
        return ['' for v in data]
    # Data for each ring
    data1 = results["strong"]   # Inner ring
    data2 = results["weak"]   # Middle ring

    # Colors (optional, for visual clarity)
    colors =  ['salmon', 'orange', 'gold', 'lightgreen', "skyblue"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.4

    # First (inner) ring
    ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.44, textprops={'fontsize': 32})

    # Second (middle) ring
    if sum(data2) > 0:
        ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.70, textprops={'fontsize': 32})

    ax.set_title(label, fontsize=16)
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)