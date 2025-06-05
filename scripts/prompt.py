# from utils import Dataset, Model,OpenAIModel
import openai
from openai import OpenAI
from litellm import completion
from vllm import LLM, SamplingParams

import time
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAIKEY")
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINIKEY")
deepseek_key = os.getenv("DEEPSEEKKEY")
# openai.api_key = os.getenv("OPENAIKEY")



def huggingface_generator(model, prompt, max_new_tokens):
    device = "cuda:0"
    hf_model, hf_tokenizer = model
    if len(hf_tokenizer.tokenize(prompt)) > hf_tokenizer.model_max_length:
        return prompt
    model_inputs = hf_tokenizer([prompt], return_tensors="pt").to(device)
    try:
        generated_ids = hf_model.generate(**model_inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    except:
        return prompt
    # generated_ids = hf_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_text = hf_tokenizer.batch_decode(generated_ids)[0]
    return generated_text


def huggingface_generator_chat(model, prompt, max_new_tokens):
    device = "cuda:0"
    hf_model, hf_tokenizer = model
    messages = [
        {"role": "system", "content": "I want you to act as a code executor. I will give you a piece of code and its input. You need to think step by step and then print the output of code execution."},
        {"role": "user", "content": prompt}
    ]
    text = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = hf_tokenizer([text], return_tensors="pt").to(device)
    generated_ids = hf_model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature = 0,
            do_sample=False,
            num_beams=1
        )
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    generated_text = hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def generator_lllm(model, prompt):
    if model == "o4-mini-2025-04-16":
        temperature = 1
    else:
        temperature = 0.0
    response = completion(
        model=model, 
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    generated_text = response.choices[0]['message']['content']
    
    return generated_text


def generator_dsr1(prompt):
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are asked to simulate the execution process of a piece of Python Code"},
            {"role": "user", "content": prompt},
        ],
        stream=False
        )
        return response.choices[0].message.content
    except:
        return "ERR"
def generator_gemini(model, prompt):
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text

def generator_vllm(model, prompt, max_tokens):
    prompts = [prompt]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_tokens)
    try:
        outputs = model.generate(prompts, sampling_params)
        output = outputs[0]
        generate_text = output.outputs[0].text
    except:
        generate_text = "ERR"
    return generate_text