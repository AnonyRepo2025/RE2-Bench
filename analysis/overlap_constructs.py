import json
import os
import matplotlib.pyplot as plt

def load_data(model_id):
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
    
    summary = {
        "input_only": list(unique_input),
        "output_only": list(unique_output),
        "overlapped": list(overlapped)
    }
    return summary
    

def count_constructs(problems_ids, constructs):
    constructs_included = []
    for i in problems_ids:
        if i not in constructs: continue
        for k in constructs[i]:
            if constructs[i][k]:
                if k not in constructs_included:
                    constructs_included.append(k)
    return constructs_included

def main(model_id):
    construct_path = "../Results/summary/constructs.json"
    construct_information = json.load(open(construct_path, 'r'))
    overlap_summary = load_data(model_id)
    
    constructs_input = count_constructs(overlap_summary["input_only"], construct_information)
    constructs_output = count_constructs(overlap_summary["output_only"], construct_information)
    constructs_overlapped = count_constructs(overlap_summary["overlapped"], construct_information)
    
    print("Input: ", constructs_input)
    print("Output: ", constructs_output)
    print("Overlapped:", constructs_overlapped)    

main("deepseek-coder-33b-instruct")