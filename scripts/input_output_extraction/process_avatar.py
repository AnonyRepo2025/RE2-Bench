import os
import json
import ast 

def smart_cast(text: str):
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text

def process_avatar():
    root = "../dataset/original_datasets/Avatar"
    wr_root = "../dataset/re2-bench/input-output/Avatar"
    if not os.path.exists(wr_root):
        os.makedirs(wr_root)
    for d in os.listdir(root):
        jsonl_path = os.path.join(wr_root, f"{d}.jsonl")
        input_dir = os.path.join(root, d, "all_tests", "input")
        objs = []
        for j in os.listdir(input_dir):
            input_path = os.path.join(input_dir, j)
            output_path = input_path.replace("input", "output")
            input_text = open(input_path, 'r').read().strip("\n")
            input_content = smart_cast(input_text)
            output_text = open(output_path, 'r').read().strip("\n")
            output_content = smart_cast(output_text)
            input_output_json = {
                "input": input_content,
                "output": output_content
            }
            objs.append(input_output_json)
        with open(jsonl_path, 'w') as f:
            for obj in objs:
                json.dump(obj, f)
                f.write('\n')
if __name__ == "__main__":
    process_avatar()