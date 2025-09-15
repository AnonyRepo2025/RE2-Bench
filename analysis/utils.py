import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 3
    
    

def donuts_distribution_with_label(results, save_path):
    plt.cla()

    def make_labels(data):
        return [str(v) if v > 3 else '' for v in data]

    data1 = results["strong"]   # inner ring
    data2 = results["weak"]     # outer ring

    colors = ['salmon', 'salmon','orange', 'orange', 'gold', 'gold',
              'lightgreen', 'lightgreen', 'skyblue', 'skyblue']
    hatches = ["", "o", "", "o", "", "o", "", "o", "", "o"]

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
    for w, h in zip(wedges, hatches):
        w.set_hatch(h)

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
        for w, h in zip(wedges2, hatches):
            w.set_hatch(h)
    

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

def donuts_distribution_without_label(results, save_path):
    plt.cla()
    def make_labels(data):
        return ['' for v in data]
    # Data for each ring
    data1 = results["strong"]   # Inner ring
    data2 = results["weak"]   # Middle ring

    # Colors (optional, for visual clarity)
    colors = ['salmon', 'salmon','orange', 'orange', 'gold', 'gold', 'lightgreen', 'lightgreen' ,"skyblue", 'skyblue']
    hatches = ["", "|||", "", "|||","", "|||","", "|||", "", "|||"]
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
    # Second (outer) ring
    if sum(data2) > 0:
        patches = ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.76, textprops={'fontsize': 40})
        for i in range(len(patches[0])):
            patches[0][i].set(hatch = hatches[i])

    plt.savefig(save_path,dpi=300)
    
    
def donuts_overlap(results, save_path):
    plt.cla()
    def make_labels(data):
        return [str(v) if v > 0 else '' for v in data]
    # Data for each ring
    data1 = results["high"]   # Inner ring
    data2 = results["low"]   # Middle ring

    # Colors (optional, for visual clarity)
    colors =  ['plum', "limegreen", "navajowhite"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.45

    # First (inner) ring
    ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.51, textprops={'fontsize': 40})

    # Second (middle) ring
    if sum(data2) > 0:
        ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.63, textprops={'fontsize': 40})

    plt.savefig(save_path,dpi=500)
