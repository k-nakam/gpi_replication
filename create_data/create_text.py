"""
Script to create texts using Llama3 model
Author: Kentaro Nakamura (knakamura@g.harvard.edu)

Args:
- prompt_file: str, directory name to the dataset of prompts (should be csv file named f"{prompt_file}.csv")
- prompt_column: str, the name of columns that contains prompt texts (default: "prompt")
- access_token: str, Huggingface access token
- start: int, the starting index of the prompts (default: None)
- end: int, the ending index of the prompts (default: None)
- task type: str, the type of task to be performed (default: "create")
    - If "create", create text based on prompts (default).
    - If "repeat", repeat the input text.
    - Otherwise, you can specify the instruction.
- cache_dir: str, directory to cache the model
- max_new_tokens: int, number of tokens to be generated (default: 100)
- save_hidden: str, directory to output the hidden states
- checkpoint: str, model checkpoint (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- hidden_use: str, specify which hidden states to be used (either "only_first", "except_first", or "all")
    - "only_first": only use the last hidden states of the first token (it contains semantic information of the entire sentences)
    - "except_first": only use the last hidden states of the tokens except the first token (not recommended)
    - "all": use the last hidden states of all tokens
    - "only_last": only use the last hidden states (recommended for decoder-only models)
- save_name: str, filename to be saved

CAUTION:
This script is compatible only with Llama3 model. You can use other models,
but you need to adjust codes (especially inputs) accordingly.
Please check each models' documentation for more details.

HOW TO USE:
To use Llama3, you need to have an access to the model (otherwise you will encounter the error "Cannot access gated repo for url"). 
You can request the access from the following url: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Once you get access to the model, you also need to log in the huggingface.
You can log in by generating access token. To generate access token, go to "profile" -> "Settings" -> "Access Token".
See the following url for more details about access token: https://huggingface.co/docs/hub/en/security-tokens

You can pass your access token to this script in two ways:
1. Set your generated access token as a parameter of this script (access_token).
2. Add your access token to the environment variable (ACCESS_TOKEN). You can create ".env" file and add "ACCESS_TOKEN=<your_access_token>".

Computational Environment:
In order to exactly replicate the original results, you need to use the same computational environment as the original paper.
While the results mostly hold, the smaller differences in the computational environment may lead to slightly different results.

The original paper used the following environment:
- GPU: NVIDIA A100 (gpu partition at Harvard FASRC) 4 GPUs
- Python: 3.10.9
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import pandas as pd
import argparse

import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--prompt_file", type=str, help='file name of prompts (should be csv file)')
parser.add_argument('--prompt_column', type=str, default='prompt', help='column name of the prompt')
parser.add_argument('--access_token', type=str, default=None, help='Huggingface access token')
parser.add_argument('--start', type=int, default=None, help='start index of the prompts (default: None)')
parser.add_argument('--end', type=int, default=None, help='end index of the prompts (default: None)')
parser.add_argument('--task_type', type=str, default='create',
                    help="""
                        If "create", create text based on prompts (default).
                        If "repeat", repeat the input text.
                        Otherwise, you can specify the instruction.
                    """)
parser.add_argument('--cache_dir', type=str, default='data/cache', help='directory to save cache')
parser.add_argument('--max_new_tokens', type=int, default=1000, help="Number of tokens to be generated")
parser.add_argument('--save_hidden', type=str, default='data/hidden_states', help='Number of texts creating')
parser.add_argument('--checkpoint', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='model checkpoint')
parser.add_argument('--hidden_use', type=str, default='only_last', help="""
                    Specify which hidden states to be used (either "only_first", "except_first", or "all")
                    - "only_first": only use the last hidden states of the first token (it contains semantic information of the entire sentences)
                    - "except_first": only use the last hidden states of the tokens except the first token (not recommended)
                    - "all": use the last hidden states of all tokens
                    - "only_last": only use the last hidden states (recommended for decoder-only models)
                    """)
parser.add_argument('--save_name', type=str, default="data/text_data", help='filename to be saved')
args = parser.parse_args()

torch.manual_seed(42)

def load_model(checkpoint: str, token: str, cache_dir: str = None) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the model from the checkpoint.

    Args:
    - checkpoint: str, model checkpoint

    Notes:
    - If you want to load different models, you might need to replace the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = token, cache_dir = cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token = token,
        cache_dir = cache_dir,
    )
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model


def get_instruction(task_type: str) -> str:
    """
    Get the instruction based on the task type.

    Args:
    - task_type: str, the type of task to be performed (either "create", "repeat", or some user specific task)

    Output:
    - instruction: str, instruction for the model
    """
    if task_type == 'create':
        #Task: create texts
        instruction = "You are a text generator who always produces the texts suggested by the prompts."
    elif task_type == 'repeat':
        #Task: repeat texts
        instruction = "You are a text generator who just repeats the input text."
    else:
        #User specific task
        print(f"Your task is: {task_type}")
        instruction = task_type
    return instruction

def load_prompts(prompt_file: str, prompt_column: str, start: int = None, end: int = None) -> list[str]:
    """
    Load the prompts from the CSV file.

    Args:
    - prompt_file: str, directory name of prompts (should be csv file)
    - prompt_column: str, the column name of the prompt
    - start: int, start index of the prompts (default: None)
    - end: int, end index of the prompts (default: None)

    Output:
    - prompts: list[str], list of prompts
    """
    try:
        if start is not None and end is not None:
            prompts = pd.read_csv(f"{prompt_file}.csv")[f"{prompt_column}"][start:end].tolist()
        else:
            prompts = pd.read_csv(f"{prompt_file}.csv")[f"{prompt_column}"].tolist()
    except Exception as e:
        logging.error(f"Failed to load prompts: {e}")
        raise
    return prompts

def generate_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    instruction: str,
    prompts: list,
    max_new_tokens: int,
    hidden_use: str,
    save_hidden: str,
    start: int = None,
    end: int = None,
    ) -> list[str, list[int]]:
    """
    Generate text based on the prompts and save the hidden states.

    Args:
    - tokenizer: AutoTokenizer, tokenizer for the model
    - model: AutoModelForCausalLM, model for text generation
    - instruction: str, instruction for the model
    - prompts: list[str], list of prompts
    - max_new_tokens: int, number of tokens to be generated
    - hidden_use: str, how to use hidden states (either "only_first", "except_first", or "all")
        - "only_first": only use the last hidden states of the first token (it contains semantic information of the entire sentences)
        - "except_first": only use the last hidden states of the tokens except the first token (not recommended)
        - "all": use the last hidden states of all tokens
    - save_hidden: str, directory to output the hidden states
    - start: int, start index of the prompts (default: None)
    - end: int, end index of the prompts (default: None)

    Output:
    - generated_texts: list[str], list of generated texts
    """
    generated_texts = []
    
    index_k = range(start, end) if start is not None and end is not None else range(len(prompts))
    
    for k, prompt in zip(index_k, prompts):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
        input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict = True,
        ).to(model.device)
        
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id= [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id = tokenizer.eos_token_id,
            do_sample=False, #for deterministic decoding
            return_dict_in_generate=True,
            output_hidden_states = True
        )
        
        if hidden_use == "only_first" or hidden_use == "all":
            #saving the first hidden states (it typically contains semantic information of the entire sentences for encoder only model)
            hidden_first = outputs.hidden_states[0][-1].view(-1, 4096) #extract the first token's hidden states from the last layer
            torch.save(hidden_first, f"{save_hidden}/hidden_first_{k}.pt")

        if hidden_use == "except_first" or hidden_use == "all":
            #saving the hidden states except the first token (it contains sequential information)    
            hidden_except_first = torch.stack([item[-1] for item in outputs.hidden_states[1:]]).view(-1, 4096)
            torch.save(hidden_except_first, f"{save_hidden}/hidden_{k}.pt")
        
        if hidden_use == 'only_last':
            #saving the last hidden states
            hidden_last = outputs.hidden_states[-1][-1].view(-1, 4096)
            torch.save(hidden_last, f"{save_hidden}/hidden_last_{k}.pt")

        #decode the generated tokens back to texts
        response = outputs.sequences[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True)
        print(f"Generated text for prompt {k}: {text}")
        generated_texts.append(text)
    return generated_texts, index_k

def save_generated_texts(generated_texts: list, prompts: list, save_name: str, index: list = None):
    """
    Save the generated texts to a pickle file.

    Args:
    - generated_texts: list[str], list of generated texts
    - prompts: list[str], list of prompts
    - save_name: str, filename to be saved
    """
    try:
        if index is not None:
            min_ind = min(index); max_ind = max(index)
            pd.DataFrame({"X": generated_texts, "P": prompts, "index": index}).to_pickle(f'{save_name}_{min_ind}_{max_ind}.pkl')
        else:
            pd.DataFrame({"X": generated_texts, "P": prompts}).to_pickle(f'{save_name}.pkl')
    except Exception as e:
        logging.error(f"Failed to save generated texts: {e}")
        raise

if __name__ == '__main__':
    load_dotenv() #take environment variables from .env.
    #log in the huggingface    
    access_token = args.access_token or os.getenv("ACCESS_TOKEN")
    if access_token is None:
        raise RuntimeError("No Hugging Face token provided.")
    print("access token:", access_token)
    
    #loading index range
    if args.start is not None and args.end is not None:
        logging.info(f"Generating texts from index {args.start} to {args.end}")
        

    #load the model
    tokenizer, model = load_model(checkpoint = args.checkpoint, token = access_token, cache_dir= args.cache_dir)

    #get the instruction
    instruction = get_instruction(task_type = args.task_type)

    #load the prompts
    prompts = load_prompts(prompt_file= args.prompt_file, prompt_column = args.prompt_column, start = args.start, end = args.end)

    #generate texts
    generated_texts, index = generate_text(
        tokenizer= tokenizer, model = model, instruction = instruction,
        prompts = prompts, max_new_tokens = args.max_new_tokens,
        hidden_use = args.hidden_use, save_hidden = args.save_hidden,
        start = args.start, end = args.end,
    )

    #save the generated texts
    save_generated_texts(generated_texts, prompts, args.save_name, index = index)
    logging.info("Finished creating texts!")