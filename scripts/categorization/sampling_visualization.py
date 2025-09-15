import json
from plot_utils import donuts_two_rings
def count_frequency(lst):
    n_swe = lst.count('Swebench')
    n_class = lst.count("Classeval")
    n_heval = lst.count("HumanEval")
    n_crux = lst.count("cruxeval")
    n_avatar = lst.count("Avatar")
    
    return [n_swe, n_class, n_avatar, n_crux, n_heval]

def read_result():
    with open("./categorization/sampled_problems.json", "r") as f:
        data = json.load(f)
        difficult_list = [data['difficult'][k]["benchmark"] for k in data["difficult"]]
        easy_list = [data['easy'][k]["benchmark"] for k in data["easy"]]
        
        counts_difficult = count_frequency(difficult_list)
        counts_easy = count_frequency(easy_list)
        
        results = {
        'dataset':['swebench', 'classeval', 'avatar', 'cruxeval', 'humaneval'],
        'strong': counts_difficult,
        'weak': counts_easy
        }
        
        save_path = "./categorization/sampled_problems.jpeg"
        donuts_two_rings(results, "", save_path)

read_result()
        