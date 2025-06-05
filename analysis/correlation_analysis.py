import csv
import json 
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
import statistics
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
import os

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

def read_json(json_path):
    json_data = json.load(open(json_path, 'r'))
    return json_data

def load_complexity():
    results = {}
    path = "../../anonymousrepo/csv_files/all_benchmarks.csv"
    data = read_csv(path)
    for d in data: 
        if 'HumanEval' in d[0]:
            pid = d[0].split("/")[-1].rstrip(".py")
        elif 'cruxeval' in d[0]:
            pid = 'cruxeval@' + d[0].split("/")[-1].rstrip(".py")
        elif 'Avatar' in d[0]:
            pid = 'Avatar@@' + d[0].split("/")[-2]
        elif 'ClassEval' in d[0]:
            pid = d[0].split("/")[-2]
        else:
            pid = d[0].split("/")[-1].rstrip(".py")
        results[pid] = d[1:10]
    return results

def load_llm(model_id):
    '''return two lists: lebels, complexity'''
    summary = {"c1": {}, "c2": {}, "c3": {}, "c4": {}, "c5": {}, "c6": {}, "c7": {}, "c8": {}, "c9": {}}
    result_path = f"../Results/summary/output_{model_id}.json"
    results = read_json(result_path)
    complexity_data = load_complexity()
    for r in results:
        if r in complexity_data:
            for i in range(len(complexity_data[r])):
                key = f"c{i+1}"
                value = complexity_data[r][i]
                
                if "complexity" not in summary[key]:
                    summary[key]["complexity"] = []
                    summary[key]["results"] = []
                summary[key]["results"].append(results[r])
                summary[key]["complexity"].append(int(value))

    return summary

def convert_data(data):
    data_list = []
    for k in data:
        if data[k][0]/(data[k][0] + data[k][1]) < 1:
            data_list.append((int(k), data[k][0]/(data[k][0] + data[k][1])))
        else:
            data_list.append((int(k), 0.5))
    data_list.sort(key=lambda s: s[0])
    return data_list


def compute_corr(model_id, label):
    data_o4mini = load_llm(model_id)[label]
    corr, p_value = pointbiserialr(data_o4mini["results"], data_o4mini["complexity"])
    return round(float(corr),2)
    

def main():
    results_dict = {
        "o4-mini-2025-04-16":[],
        "gemini-2.5-pro-preview-03-25": [],
        "deepseek-r1": [],
        "gemini-1.5-pro": [],
        "gpt-4.1-2025-04-14": [],
        "deepseek-coder-33b-instruct": []
    }
    for i in range(1, 10):
        label = f"c{i}"
        corr1 = compute_corr("o4-mini-2025-04-16", label)
        results_dict["o4-mini-2025-04-16"].append(corr1)
        corr2 = compute_corr("gemini-2.5-pro-preview-03-25", label)
        results_dict["gemini-2.5-pro-preview-03-25"].append(corr2)
        corr3 = compute_corr("deepseek-r1", label)
        results_dict["deepseek-r1"].append(corr3)
        corr4 = compute_corr("gemini-1.5-pro", label)
        results_dict["gemini-1.5-pro"].append(corr4)
        corr5 = compute_corr("gpt-4.1-2025-04-14", label)
        results_dict["gpt-4.1-2025-04-14"].append(corr5)
        corr6 = compute_corr("deepseek-coder-33b-instruct", label)
        results_dict["deepseek-coder-33b-instruct"].append(corr6)
    
    json_path = "../Results_correlation_output.json"
    if not os.path.exists(json_path):
        with open(json_path, 'w') as wr:
            json.dump(results_dict, wr, indent=4)
main()