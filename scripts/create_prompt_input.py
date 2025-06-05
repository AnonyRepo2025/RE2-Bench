with open("prompts_icl_examples/input_prediction/general.txt", "r", encoding="utf-8") as file:
    icl_example = file.read()

with open("prompts_icl_examples/input_prediction/classeval.txt", "r", encoding="utf-8") as file:
    icl_example_classeval = file.read()

with open("prompts_icl_examples/input_prediction/human_crux.txt", "r", encoding="utf-8") as file:
    icl_example_human_crux = file.read()


def create_prompt_swebench(code, code_output, input_hinted, fun_name, dependency):
    if dependency:
        question = """
[PYTHON]
{}
[/PYTHON]
Functions called during the execution:
[PYTHON]
{}
[/PYTHON]
What will be the input of `{}`, given the following output:
[OUTPUT]
{}
[/OUTPUT]
[INPUT]
{}
[/INPUT]
[THOUGHT]
        """
        question = question.format(code, dependency, fun_name, code_output, input_hinted)
    else:
        question = """
[PYTHON]
{}
[/PYTHON]
Functions called during the execution:
What will be the input of `{}`, given the following output:
[OUTPUT]
{}
[/OUTPUT]
[INPUT]
{}
[/INPUT]
[THOUGHT]
        """        
        question = question.format(code, fun_name, code_output, input_hinted)
    prompt = icl_example + question    
    return prompt 

def create_prompt_humaneval_cruxeval(code, code_output, fun_name):
    question = """
[PYTHON]
{}
[/PYTHON]
What will be the output of the code, given the following output:
[OUTPUT]
{}
[/OUTPUT]
[INPUT]
{}("")
[/INPUT]
[THOUGHT]
"""
    question = question.format(code, code_output, fun_name)
    prompt = icl_example_human_crux + question
    return prompt


def create_prompt_avatar(code, code_output):
    question = """
[PYTHON]
{}
[/PYTHON]
What will be the output of the code, given the following output:
[OUTPUT]
{}
[/OUTPUT]
[INPUT]
""
[/INPUT]
[THOUGHT]
"""
    question = question.format(code, code_output)
    prompt = icl_example + question
    return prompt

def create_prompt_classeval(code, output, input_hint):
    question = """
[PYTHON]
{}
[/PYTHON]
What will be the input of the `Test.test()`, given the following output:
[OUTPUT]
{}
[/OUTPUT]

[INPUT]
{}
[INPUT]
[THOUGHT]
"""
    question = question.format(code, output, input_hint)
    prompt = icl_example_classeval + question
    return prompt

def create_prompt_deepseekcoder(prompt):
    wrapped = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n" + "### Instruction:\n" + prompt + "\n### Response:"
    return wrapped