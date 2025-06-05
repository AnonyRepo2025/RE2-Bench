import pandas as pd
import matplotlib.pyplot as plt

# Define mapping of CSV file paths to figure labels
Names = {
    "existing/avatar_complexity.csv": "(a) Avatar",
    "existing/classeval_complexity.csv": "(b) ClassEval",
    "existing/cruxeval_complexity.csv": "(c) CRUXEval",
    "existing/humaneval_complexity.csv": "(d) HumanEval",
    "existing/swebench_complexity.csv": "(e) SWE-bench",
    # "fff.csv": "(e) SWE-Bench",
}

def plot_box_charts(csv_paths):
    original_columns = [
        "Base complexity", 
        "Predicates with operators", 
        "Nested levels", 
        "Complex code structures", 
        "Third-Party calls", 
        "Inter_dependencies", 
        "Intra_dependencies",
        "R8",
        "R9"
    ]
    
    # Define new column names for readability (C1, C2, ...)
    new_columns = [f'M$_{str(i + 1)}$' for i in range(len(original_columns))]
    
    # Set layout: 5 subfigures in a row
    rows = 1  # Single row
    cols = 5  # Five subfigures in a row

    # Adjust figure size to balance readability and compactness
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3))
    axs = axs.flatten()
    

    for idx, (ax, csv_path) in enumerate(zip(axs, csv_paths)):
        df = pd.read_csv(csv_path)
        
        # Rename columns for readability
        rename_dict = dict(zip(original_columns, new_columns))
        
        df.rename(columns=rename_dict, inplace=True)
        # print(df)

        # Create a boxplot
        boxplot = df[new_columns].boxplot(patch_artist=True, return_type='dict', ax=ax)
        colors = ['#a0c4ff', '#b5ea8c', '#d6e6ff', '#ffd6a5', '#fbe0e0', '#bdb2ff', 'pink', "grey", "thistle"]
        ax.grid(False)
        
        for i, color in enumerate(colors):
            plt.setp(boxplot['boxes'][i], color=color, facecolor=color)
            plt.setp(boxplot['medians'][i], color="#525e75", linewidth=2)
            plt.setp(boxplot['fliers'][i], marker='o', color="grey", markersize=3)
            plt.setp(boxplot['whiskers'][2*i:2*i+2], color="black")
            plt.setp(boxplot['caps'][2*i:2*i+2], color="black")

        # Set x-axis label with larger font (subfigure label)
        ax.set_xlabel(Names[csv_path], fontsize=20)
        ax.tick_params(axis='x', labelsize=15)

        # Show y-axis tick labels for the leftmost subplot and the swe-bench subplot
        if idx == 0:
            ax.set_ylabel('Metric Value', fontsize=20)
            ax.tick_params(axis='y', labelsize=16)
        elif "swebench_complexity.csv" in csv_path or 'fff.csv' in csv_path:
            # Display y-axis tick labels for swe-bench to show the full 0-100 range
            ax.tick_params(axis='y', labelsize=17)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        # Adjust y-limits for different datasets
        if "swebench_complexity.csv" in csv_path or 'fff.csv' in csv_path:
            ax.set_ylim(0, 40)
        else:
            ax.set_ylim(0, 20)

    # Reduce the gap between subfigures by setting a smaller horizontal padding
    plt.tight_layout(w_pad=0.5)
    plt.savefig("../figs/motivating_example.pdf")

# Example usage:
plot_box_charts([
    "existing/avatar_complexity.csv", 
    "existing/classeval_complexity.csv",
    "existing/cruxeval_complexity.csv",
    "existing/humaneval_complexity.csv",  
    "existing/swebench_complexity.csv"
    # 'fff.csv'
])
