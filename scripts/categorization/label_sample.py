import csv
import pandas as pd
from plot_utils import donuts, donuts_two_rings
import argparse
from silhouette_dbi import Silhouette_DBI_three_category, Silhouette_DBI_two_category

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


def get_quartiles(cols, p1, p2, p3, p4):
    series = pd.Series(cols)
    quantiles = series.quantile([p1, p2, p3, p4])
    return quantiles.to_dict()


def tag_dataset(raw_dataset, quantile_dict, start_idx, end_idx, p1, p2, p3, p4):
    cleaned_dataset = []
    count = 0
    # print(quantile_dict)
    for data in raw_dataset:
        new_data = []
        for i in range(len(data)):
            if i not in list(range(start_idx, end_idx)):
                new_data.append(data[i])
            else:
                quantiles = quantile_dict[i]
                ## categorize items with computed quantiles
                if int(data[i]) > quantiles[p4]:
                    new_data.append("strong")
                elif int(data[i]) > quantiles[p2] and int(data[i]) <= quantiles[p3]:
                    new_data.append("medium")
                elif int(data[i]) <= quantiles[p1]:
                    new_data.append("weak")
                else:
                    new_data.append("other")
        cleaned_dataset.append(new_data)
    return cleaned_dataset                 



def count_dataset(li):
    n_swe = li.count('swebench')
    n_cla = li.count('classeval')
    n_ava = li.count('avatar')
    n_cru = li.count('cruxeval')
    n_hum = li.count('humaneval')
    
    return [n_swe, n_cla, n_ava, n_cru, n_hum]


def parse_dataset_from_string(string):
    dataset = ""
    if 'Avatar' in string:
        dataset = 'avatar'
    elif 'classeval' in string:
        dataset = 'classeval'
    elif 'cruxeval' in string:
        dataset = 'cruxeval'
    elif 'HumanEval' in string:
        dataset = 'humaneval'
    elif 'swebench' in string:
        dataset = 'swebench'
    return dataset
          
def tag_attributes(csv_path, p1, p2, p3, p4):
    raw_data = read_csv(csv_path)
    start_idx, end_idx = 1, 10
    quantile_dict = {}
    ## iterate through indexes to get quantiles of different columns
    for i in range(start_idx, end_idx):
        cols = [int(rd[i]) for rd in raw_data]
        quantiles = get_quartiles(cols, p1, p2, p3, p4)
        quantile_dict[i] = quantiles
        print(f"M{i}: {quantiles}")

    tagged_data = tag_dataset(raw_data, quantile_dict, start_idx, end_idx, p1, p2, p3, p4)
    csv_path_wr = f"./categorization/all_benchmarks-tagged_{p1}_{p2}_{p3}_{p4}.csv"
    with open(csv_path_wr, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(tagged_data)

def tag_samples(knob, p1, p2, p3, p4):
    file_locations = {
        "strong": {},
        "weak":{},
        "medium":{},
        "other": {}
    }
    
    difficulty_dict = {}
    
    list_strong = []
    list_medium = []
    list_weak = []
    csv_path = f"./categorization/all_benchmarks-tagged_{p1}_{p2}_{p3}_{p4}.csv"
    csv_data = read_csv(csv_path)
    for data in csv_data:
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
        tags = data[1:10]
        n_strong = tags.count('strong')
        n_medium = tags.count('medium')
        n_weak = tags.count('weak')
        if n_strong >= knob:
            difficulty_dict[data[0]] = "difficult"
            list_strong.append(dataset)
            if dataset not in file_locations["strong"]:
                file_locations["strong"][dataset] = []
            if data[0] not in file_locations["strong"][dataset]:
                file_locations["strong"][dataset].append(data[0])
        elif n_medium >= knob:
            difficulty_dict[data[0]] = "medium"
            list_medium.append(dataset)
            if dataset not in file_locations["medium"]:
                file_locations["medium"][dataset] = []
            if data[0] not in file_locations["medium"][dataset]:
                file_locations["medium"][dataset].append(data[0])
        elif n_weak >= knob:
            difficulty_dict[data[0]] = "easy"
            list_weak.append(dataset)
            if dataset not in file_locations["weak"]:
                file_locations["weak"][dataset] = []
            if data[0] not in file_locations["weak"][dataset]:
                file_locations["weak"][dataset].append(data[0])
    
    strong_info = count_dataset(list_strong)
    medium_info = count_dataset(list_medium)
    weak_info = count_dataset(list_weak)
    print(f"len(strong):{len(list_strong)}")
    print(f"len(medium):{len(list_medium)}")
    print(f"len(weak):{len(list_weak)}")
    
    results = {
        'dataset':['swebench', 'classeval', 'avatar', 'cruxeval', 'humaneval'],
        'strong': strong_info,
        'medium': medium_info,
        'weak': weak_info
        }
    print(results)
    # save_path = f"./categorization/difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.jpeg"
    # donuts(results, 'c10', save_path)
    return difficulty_dict

def create_csv_with_tags(knob, p1, p2, p3, p4, difficulty_dict):
    csv_with_tag_path = f"./categorization/all_benchmarks-difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.csv"
    csv_path = "./categorization/all_benchmarks.csv"
    raw_data = read_csv(csv_path)
    
    with open(csv_with_tag_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'dataset', 'difficulty', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'])
        for data in raw_data:
            data_index = data[0]
            if data_index in difficulty_dict:
                row = []
                dataset = parse_dataset_from_string(data_index)
                row.append(data_index)
                row.append(dataset)
                row.append(difficulty_dict[data_index])
                properties = data[1: 10]
                for p in properties:
                    row.append(p)
                writer.writerow(row)


def remove_borderlines(borderline_examples, p1, p2, p3, p4, knob):
    csv_path = f"./categorization/all_benchmarks-difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.csv"
    df = pd.read_csv(csv_path)

    df = df[df["difficulty"] != "medium"]
    df = df[~df["index"].isin(borderline_examples)]  # use isin for lists/sets
    df.to_csv(csv_path, index=False)

def create_donut_plots(p1, p2, p3, p4, knob, dbi):
    list_strong, list_easy = [], []
    csv_path = f"./categorization/all_benchmarks-difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.csv"
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            difficulty = row["difficulty"]
            if difficulty == "difficult":
                list_strong.append(dataset)
            else:
                list_easy.append(dataset)
    strong_info = count_dataset(list_strong)
    easy_info = count_dataset(list_easy)
    
    print(f"Difficult:{len(list_strong)};Easy:{len(list_easy)}")
    
    results = {
        'dataset':['swebench', 'classeval', 'avatar', 'cruxeval', 'humaneval'],
        'strong': strong_info,
        'weak': easy_info
    }
    
    save_path = f"./categorization/donuts_charts/difficulty_{p1}_{p4}@{knob}.jpeg"
    title = f"[{p1}, {p4}], knob:{knob}, dbi: {dbi}"
    donuts_two_rings(results, title, save_path)
    return min(len(list_strong), len(list_easy))        


def find_best_knob(csv_path, p1, p2, p3, p4):
    tag_attributes(csv_path, p1, p2, p3, p4)
    best_knob = -1
    max_sample_number = 0
    for knob in range(5,10):
        difficulty_dict = tag_samples(knob, p1, p2, p3, p4)
        create_csv_with_tags(knob, p1, p2, p3, p4, difficulty_dict)
        
        borderline_examples, _, silhouette_score_pre = Silhouette_DBI_two_category(p1, p2, p3, p4, knob, remove_borderlines=True)
        remove_borderlines(borderline_examples,p1, p2, p3, p4, knob)
        
        _, final_dbi, silhouette_score_after = Silhouette_DBI_two_category(p1, p2, p3, p4, knob)
        
                
        print(f"#############{p1}-{p4}-{knob}#############")
        print(silhouette_score_pre, silhouette_score_after)
        print(f"Borderlines removed: {len(borderline_examples)}")
        sample_number = create_donut_plots(p1, p2, p3, p4, knob, final_dbi)
        print(f"#############")
        
        if sample_number > max_sample_number:
            max_sample_number = sample_number
            best_knob = knob
    print(f"Best KNOB: {best_knob}")
    print(f"Number Sampled: {max_sample_number}")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=float)
    parser.add_argument("--p2", type=float, default=0)
    parser.add_argument("--p3", type=float, default =0)
    parser.add_argument("--p4", type=float)
    parser.add_argument("--knob", type=int, default="0")
    args = parser.parse_args()
    
    p1 = args.p1
    p2 = args.p2 
    p3 = args.p3 
    p4 = args.p4
    knob = args.knob
    
    csv_path = "./categorization/all_benchmarks.csv"
    
    find_best_knob(csv_path, p1, p2, p3, p4)
    
    
    
    
    