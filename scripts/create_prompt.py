with open("prompts_icl_examples/output_prediction/general.txt", "r", encoding="utf-8") as file:
    icl_example = file.read()
with open("prompts_icl_examples/output_prediction/classeval.txt", "r", encoding="utf-8") as file:
    icl_example_classeval = file.read()

def create_prompt_swebench(code, code_input, output_hinted, fun_name, dependency):
    if dependency:
        question = """
[PYTHON]
{}
[/PYTHON]
Functions called during the execution:
[PYTHON]
{}
[/PYTHON]
What will be the output of `{}`, given the following input:
[INPUT]
```{}```
[/INPUT]
[OUTPUT]
```{}```
[/OUTPUT]
[THOUGHT]
        """
        question = question.format(code, dependency, fun_name, code_input, output_hinted)
    else:
        question = """
[PYTHON]
{}
[/PYTHON]
Functions called during the execution:
What will be the output of `{}`, given the following input:
[INPUT]
```{}```
[/INPUT]
[OUTPUT]
```{}```
[/OUTPUT]
[THOUGHT]
        """        
        question = question.format(code, fun_name, code_input, output_hinted)
    prompt = icl_example + question    
    return prompt 

def create_prompt_humaneval_cruxeval(code, code_input):
    question = """
[PYTHON]
{}
[/PYTHON]
What will be the output of the code, given the following input:
[INPUT]
```{}```
[/INPUT]
[OUTPUT]
```{{"output":""}}```
[/OUTPUT]
[THOUGHT]
"""
    question = question.format(code, code_input)
    prompt = icl_example + question
    return prompt
    
def create_prompt_classeval(code):
    question = """
[PYTHON]
{}
[/PYTHON]
What will be the output of the `Test.test()`?
[OUTPUT]
```{{"output":""}}```
[/OUTPUT]
[THOUGHT]
"""
    question = question.format(code)
    prompt = icl_example_classeval + question
    return prompt

def create_prompt_deepseekcoder(prompt):
    wrapped = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n" + "### Instruction:\n" + prompt + "\n### Response:"
    return wrapped