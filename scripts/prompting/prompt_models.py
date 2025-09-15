from litellm import completion
import os
import time
from tqdm import tqdm
import argparse
from openai import OpenAI

def litellm_generator(model, prompt, max_new_tokens):
    retries = 8
    attempt = 0
    if model == "gpt-4o-mini":
        temperature = 1
    else:
        temperature = 0.0
    while attempt < retries:
        try:
            output = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0
            )
            response_text = output.choices[0]['message']['content']
            if response_text:
                print(f"Resolved after {attempt+1} attempts")
                return False, response_text
            else:
                attempt += 1
                time.sleep(2)
        except Exception as e:
            print(e)
            if 'maximum context length is' in str(e):
                output = ''
                return True, output
            else:
                time.sleep(3)
                attempt += 1
    print(f"failed after {retries} attempts")
    return True, ""

def openrouter_generator(model, prompt, max_new_tokens):
    retries = 5
    attempt = 0
    
    if model == "openai/o4-mini":
        temp = 1.0
    else:
        temp = 0.0

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key = ""## USE YOUR OWN OPENROUTER API
    )

    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp
            )
            return False, completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            time.sleep(2)  # optional backoff before retry

    # If all retries fail
    return True, ""

def llm_inference(model, task, max_tokens):
    prompt_root = f"../prompts/{task}"
    model_name = model.split("/")[-1]
    response_root = f"../results/{task}/{model_name}"
    for difficulty in os.listdir(prompt_root):
        ## difficulty: difficult or medium
        prompt_folder = os.path.join(prompt_root, difficulty)
        
        print(f"Prompting {model} on {difficulty} problems")
        for filename in tqdm(os.listdir(prompt_folder)):
            problem_index, _  = os.path.splitext(filename)
            
            file_path = os.path.join(prompt_folder, filename)
            
            write_folder = os.path.join(response_root, difficulty)
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            response_path = os.path.join(write_folder, f"{problem_index}.txt")
            if os.path.exists(response_path):
                continue
            
            with open(file_path, 'r') as f:
                prompt = f.read()
            err_flag, response = openrouter_generator(model, prompt, max_new_tokens=max_tokens)
            
            if not err_flag:
                with open(response_path, 'w') as f:
                    if response:
                        f.write(response)
                    else:
                        print("Error in problem ", problem_index)
                        f.write("Error")
            else:
                print("Error in problem ", problem_index)
                with open(response_path, 'w') as f:
                    f.write("Error")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    args = parser.parse_args()
    model = args.model
    task = args. task
    max_tokens = args.max_tokens
    llm_inference(model, task, max_tokens)