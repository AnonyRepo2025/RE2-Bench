from promptFactory import OutputPredictionPrompt, InputPredictionPrompt, OutputPredictionWoHintPrompt, InputPredictionWoHintPrompt, InputPredictionWoCoTPrompt, OutputPredictionWoCoTPrompt
import json
import os
from utils import read_sampling_results, read_code, read_io,read_dependency, mask_input, mask_output


def create_output_prediction_prompts():
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = json.dumps(io["input"], indent=4) ## pretty print the json
        code_output = io["output"]
        entry_point = io["entry"]
        
        output_structure = mask_output(code_output)
        output_prompts = OutputPredictionPrompt()
        if source == "HumanEval" or source == "cruxeval":
            prompt = output_prompts.humanEvalPrompt(code, entry_point, code_input, output_structure)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            prompt = output_prompts.swebenchPrompt(code, entry_point, code_input, output_structure, dep_code)
        elif source == "Avatar":
            code_input = json.dumps({"input": code_input}, indent=4)
            prompt = output_prompts.avatarPrompt(code, entry_point, code_input, output_structure)
        elif source == "Classeval":
            prompt = output_prompts.classevalPrompt(code, entry_point, code_input, output_structure)
        
        ## write prompts to txt files
        wr_root = f"../prompts/output_prediction/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)
            
def create_input_prediction_prompts():        
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = io["input"]
        code_output = io["output"]
        entry_point = io["entry"]
        
        input_structure = mask_input(code_input)
        # print(input_structure)
        
        input_prompts = InputPredictionPrompt()
        if source == "HumanEval" or source == "cruxeval":
            code_output = json.dumps({"output": code_output}, indent=4)
            prompt = input_prompts.humanEvalPrompt(code, entry_point, code_output, input_structure)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.swebenchPrompt(code, entry_point, code_output, input_structure, dep_code)
        elif source == "Avatar":
            code_output = json.dumps({"output": code_output}, indent=4)
            code_input = json.dumps({"input": code_input}, indent=4)
            input_structure = json.dumps({"input": "XXX"}, indent=4).replace('"XXX"', 'XXX')
            prompt = input_prompts.avatarPrompt(code, entry_point, code_output, input_structure)
        elif source == "Classeval":
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.classevalPrompt(code, entry_point, code_output, input_structure)

        ## write prompts to txt files
        wr_root = f"../prompts/input_prediction/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)

def create_output_prediction_wohint_prompts():
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = json.dumps(io["input"], indent=4) ## pretty print the json
        code_output = io["output"]
        entry_point = io["entry"]
        
        output_prompts = OutputPredictionWoHintPrompt()
        if source == "HumanEval" or source == "cruxeval":
            prompt = output_prompts.humanEvalPrompt(code, entry_point, code_input)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            prompt = output_prompts.swebenchPrompt(code, entry_point, code_input, dep_code)
        elif source == "Avatar":
            code_input = json.dumps({"input": code_input}, indent=4)
            prompt = output_prompts.avatarPrompt(code, entry_point, code_input)
        elif source == "Classeval":
            prompt = output_prompts.classevalPrompt(code, entry_point, code_input)
        
        ## write prompts to txt files
        wr_root = f"../prompts/output_prediction_wohint/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)


def create_input_prediction_wohint_prompts():        
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = io["input"]
        code_output = io["output"]
        entry_point = io["entry"]
          
        input_prompts = InputPredictionWoHintPrompt()
        if source == "HumanEval" or source == "cruxeval":
            code_output = json.dumps({"output": code_output}, indent=4)
            prompt = input_prompts.humanEvalPrompt(code, entry_point, code_output)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.swebenchPrompt(code, entry_point, code_output, dep_code)
        elif source == "Avatar":
            code_output = json.dumps({"output": code_output}, indent=4)
            code_input = json.dumps({"input": code_input}, indent=4)
            prompt = input_prompts.avatarPrompt(code, entry_point, code_output)
        elif source == "Classeval":
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.classevalPrompt(code, entry_point, code_output)

        ## write prompts to txt files
        wr_root = f"../prompts/input_prediction_wohint/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)


def create_output_prediction_wocot_prompts():
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = json.dumps(io["input"], indent=4) ## pretty print the json
        code_output = io["output"]
        entry_point = io["entry"]
        
        output_structure = mask_output(code_output)
        output_prompts = OutputPredictionWoCoTPrompt()
        if source == "HumanEval" or source == "cruxeval":
            prompt = output_prompts.humanEvalPrompt(code, entry_point, code_input, output_structure)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            prompt = output_prompts.swebenchPrompt(code, entry_point, code_input, output_structure, dep_code)
        elif source == "Avatar":
            code_input = json.dumps({"input": code_input}, indent=4)
            prompt = output_prompts.avatarPrompt(code, entry_point, code_input, output_structure)
        elif source == "Classeval":
            prompt = output_prompts.classevalPrompt(code, entry_point, code_input, output_structure)
        
        ## write prompts to txt files
        wr_root = f"../prompts/output_prediction_wocot/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)


def create_input_prediction_wocot_prompts():        
    ## load sampled problems
    samples = read_sampling_results()
    for sample in samples:
        difficulty = sample["difficulty"]
        sample_id = sample["id"]
        source = sample["source"]
        io_id = sample["io_id"]
        
        code = read_code(sample_id, source)
        io = read_io(sample_id, source, io_id)
        
        code_input = io["input"]
        code_output = io["output"]
        entry_point = io["entry"]
        
        input_structure = mask_input(code_input)
        # print(input_structure)
        
        input_prompts = InputPredictionWoCoTPrompt()
        if source == "HumanEval" or source == "cruxeval":
            code_output = json.dumps({"output": code_output}, indent=4)
            prompt = input_prompts.humanEvalPrompt(code, entry_point, code_output, input_structure)
        elif source == "Swebench":
            dep_code = read_dependency(sample_id)
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.swebenchPrompt(code, entry_point, code_output, input_structure, dep_code)
        elif source == "Avatar":
            code_output = json.dumps({"output": code_output}, indent=4)
            code_input = json.dumps({"input": code_input}, indent=4)
            input_structure = json.dumps({"input": "XXX"}, indent=4).replace('"XXX"', 'XXX')
            prompt = input_prompts.avatarPrompt(code, entry_point, code_output, input_structure)
        elif source == "Classeval":
            if not isinstance(code_output, dict):
                code_output = {"output": code_output}
            code_output = json.dumps(code_output, indent=4)
            prompt = input_prompts.classevalPrompt(code, entry_point, code_output, input_structure)

        ## write prompts to txt files
        wr_root = f"../prompts/input_prediction_wocot/{difficulty}"
        if not os.path.exists(wr_root):
            os.makedirs(wr_root)
        txt_path = os.path.join(wr_root, f"{sample_id}.txt")
        with open(txt_path, "w") as wr:
            wr.write(prompt)
            
# create_output_prediction_prompts()
if __name__ == "__main__":
    create_output_prediction_prompts()
    create_input_prediction_prompts()
    create_output_prediction_wohint_prompts()
    create_input_prediction_wohint_prompts()
    
    create_input_prediction_wocot_prompts()
    create_output_prediction_wocot_prompts()