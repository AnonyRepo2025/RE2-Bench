import json
import os
import sys
from collections import defaultdict

def load_unique_json_objects(filepath):
    unique_objs = []
    seen = set()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = json.dumps(obj, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_objs.append(obj)
    
    return unique_objs

def group_by_name(objects):
    grouped = defaultdict(list)
    for obj in objects:
        name = obj.get('name', 'unknown')
        grouped[name].append(obj)
    return grouped

def write_grouped_jsons(grouped_data, original_filename):
    base_name = os.path.splitext(original_filename)[0]
    for name, objs in grouped_data.items():
        out_file = f"{base_name}@{name}.jsonl"
        with open(out_file, 'w') as f:
            for obj in objs:
                json.dump(obj, f)
                f.write('\n')
        print(f"Written: {out_file} ({len(objs)} entries)")

def main(json_file_path):
    print(f"Processing file: {json_file_path}")
    unique_objs = load_unique_json_objects(json_file_path)
    grouped = group_by_name(unique_objs)
    write_grouped_jsons(grouped, json_file_path)
    os.remove(json_file_path)
    print("Done.")

if __name__ == "__main__":
    root_folder = sys.argv[1]
    for d in os.listdir(root_folder):
        if "@" not in d:
            json_path = os.path.join(root_folder, d)
            main(json_path)