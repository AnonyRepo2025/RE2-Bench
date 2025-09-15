import os

class OutputPredictionPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/output_prediction"
    def humanEvalPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}` given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[THOUGHT]
        """
        return icl_example + "\n" + prompt
    
    
    



class InputPredictionPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/input_prediction"
    def humanEvalPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}` given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the input of `{entry_point}`, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of the code snippet, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}`, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[THOUGHT]
        """
        return icl_example + "\n" + prompt
    
    
    
class InputPredictionWoHintPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/input_prediction_wohint"
    def humanEvalPrompt(self, code: str, entry_point: str, output_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}` given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, output_string: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the input of `{entry_point}`, given the following input:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, output_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of the code snippet, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, output_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}`, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[THOUGHT]
        """
        return icl_example + "\n" + prompt
    
    
    

class OutputPredictionWoHintPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/output_prediction_wohint"
    def humanEvalPrompt(self, code: str, entry_point: str, input_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}` given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, input_string: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, input_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[THOUGHT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, input_string: str) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[THOUGHT]
        """
        return icl_example + "\n" + prompt
    
    
    

class OutputPredictionWoCoTPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/output_prediction_wocot"
    def humanEvalPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}` given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[OUTPUT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[OUTPUT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[OUTPUT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, input_string: str, output_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the output of `{entry_point}`, given the following input:
[INPUT]
```
{input_string}
```
[/INPUT]

[STRUCTURE]
```
{output_structure}
```
[/STRUCTURE]

[OUTPUT]
        """
        return icl_example + "\n" + prompt





class InputPredictionWoCoTPrompt:
    def __init__(self):
        self.icl_root = "./prompts_icl_examples/input_prediction_wocot"
    def humanEvalPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "humaneval_cruxeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""
[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}` given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[INPUT]
"""
        return icl_example + "\n" + prompt
    
    def swebenchPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str, dependency_string: str ) -> str:
        icl_path = os.path.join(self.icl_root, "swebench.txt")
        icl_example = open(icl_path, "r").read()
        dep_info = f"\nFunctions called during the execution:\n[PYTHON]\n{dependency_string}\n[/PYTHON]" if dependency_string else ""
        prompt = f"""[PYTHON]
{code}
[/PYTHON]
{dep_info}
What will be the input of `{entry_point}`, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[INPUT]
"""
        return icl_example + "\n" + prompt
    
    def avatarPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "avatar.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of the code snippet, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[INPUT]
"""
        return icl_example + "\n" + prompt
    
    
    def classevalPrompt(self, code: str, entry_point: str, output_string: str, input_structure: str ) -> str:
        icl_path = os.path.join(self.icl_root, "classeval.txt")
        icl_example = open(icl_path, "r").read()
        prompt = f"""[PYTHON]
{code}
[/PYTHON]

What will be the input of `{entry_point}`, given the following output:
[OUTPUT]
```
{output_string}
```
[/OUTPUT]

[STRUCTURE]
```
{input_structure}
```
[/STRUCTURE]

[INPUT]
        """
        return icl_example + "\n" + prompt