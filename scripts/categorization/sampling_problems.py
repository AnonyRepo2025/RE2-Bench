import pandas as pd
import random
import json
import os

def process_csv(file_path, seed=None):

    if seed is not None:
        random.seed(seed)
    
    df = pd.read_csv(file_path)

    # 1. All indices of 'difficult'
    difficult_indices = df.loc[df["difficulty"] == "difficult", "index"].tolist()
    
    # 2. Number of 'difficult' samples
    N = len(difficult_indices)
    
    # 3. Randomly sample N from 'easy'
    easy_indices = df.loc[df["difficulty"] == "easy", "index"].tolist()
    easy_sampled_indices = random.sample(easy_indices, min(N, len(easy_indices)))
    
    return difficult_indices, N, easy_sampled_indices

def clean_indices(index_string):
    if "swebench" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "Swebench"
    elif "classeval" in index_string:
        return index_string.split("/")[-2], "Classeval"
    elif "cruxeval" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "cruxeval"
    elif "HumanEval" in index_string:
        return index_string.split("/")[-1].removesuffix(".py"), "HumanEval"
    elif "Avatar" in index_string:
        return index_string.split("/")[-2], "Avatar"

def sample_input_output(index, dataset):
    input_ouput_root = f"../dataset/re2-bench/input-output/{dataset}"
    if dataset == "Swebench":
        modified_filename = index.split("@@")[0] + "@@" + index.split("@@")[1].rstrip(".py") + "." + index.split("@@")[2] + ".jsonl"
    elif dataset == "cruxeval":
        modified_filename = index.replace("sample", "cruxeval") + ".jsonl"
    elif dataset == "HumanEval":
        modified_filename = index + ".jsonl"
    elif dataset == "Classeval":
        modified_filename = index.split("@")[-1].replace(".", "@") + ".jsonl"
    elif dataset == "Avatar":
        modified_filename = index + ".jsonl"
    file_path = os.path.join(input_ouput_root, modified_filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        ## sample an input-output pair
        with open(file_path, 'r') as f:
            number_candidates = sum(1 for _ in f)
            return random.randint(0, number_candidates - 1)
            
        

if __name__ == "__main__":
    results = {
        "difficult": {},
        "easy": {}
    }
    
    difficult_indices, N, easy_sampled_indices = process_csv("./categorization/all_benchmarks-difficulty_0.25_0.35_0.65_0.75@6.csv", seed=6)
    
    difficult_problems_ids = [clean_indices(idx)[0] for idx in difficult_indices]
    easy_problems_ids = [clean_indices(idx)[0] for idx in easy_sampled_indices]
    
    difficult_benchmarks = [clean_indices(idx)[1] for idx in difficult_indices]
    easy_benchmarks = [clean_indices(idx)[1] for idx in easy_sampled_indices]
    
    for i, b in zip(difficult_problems_ids, difficult_benchmarks):
        sampled_io = sample_input_output(i, b)
        results["difficult"][i] = {
            "benchmark": b,
            "input-output": sampled_io
        }
    
    for i, b in zip(easy_problems_ids, easy_benchmarks):
        sampled_io = sample_input_output(i, b)
        sampled_io = sample_input_output(i, b)
        results["easy"][i] = {
            "benchmark": b,
            "input-output": sampled_io
        }
    
    
    
    
    result_path = "./categorization/sampled_problems.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)