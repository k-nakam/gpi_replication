# About this folder

This folder contains the files needed to create the texts and extract the internal representations of LLM (LLaMa3-8B). This folder contains the following files:

1. `create_data/create_text.py`: Python code to generate the texts and extract the internal representations.

2. `create_data/label_text.py`: Python code used to assign the hypothetical confounding variable


Here are the arguments in `create_data/create_text.py`:

- `prompt_file`: str, directory name to the dataset of prompts (should be csv file named f"{prompt_file}.csv")

- `prompt_column`: str, the name of columns that contains prompt texts (default: "prompt")

- `access_token`: str, Huggingface access token

- `start`: int, the starting index of the prompts (default: None)

- `end`: int, the ending index of the prompts (default: None)

- `task_type`: str, the type of task to be performed (default: "create")
    - If "create", create text based on prompts (default).
    - If "repeat", repeat the input text.
    - Otherwise, you can specify the instruction.

- `cache_dir`: str, directory to cache the model

- `max_new_tokens`: int, number of tokens to be generated (default: 100)

- `save_hidden`: str, directory to output the hidden states

- `checkpoint`: str, model checkpoint (default: "meta-llama/Meta-Llama-3-8B-Instruct")

- `hidden_use`: str, specify which hidden states to be used (either "only_first", "except_first", or "all")
    - "only_first": only use the last hidden states of the first token (it contains semantic information of the entire sentences)
    - "except_first": only use the last hidden states of the tokens except the first token (not recommended)
    - "all": use the last hidden states of all tokens
    - "only_last": only use the last hidden states (recommended for decoder-only models)

- `save_name`: str, filename to be saved


## How to Use

1. To replicate simulation studies (create):

```bash
python create_data/create_text.py \
    --prompt_file "data/simulation/prompt_create" \
    --max_new_tokens 4096 \
    --save_hidden "data/simulation/hidden_create" \
    --task_type "create" \
    --start 0 \
    --end 4000 \
    --save_name "data/simulation/text_create"
```

2. To replicate simulation studies (reuse):

```bash
python create_data/create_text.py \
    --prompt_file "data/simulation/prompt_repeat" \
    --max_new_tokens 4096 \
    --task_type "repeat" \
    --save_hidden "data/simulation/hidden_reuse" \
    --start 0 \
    --end 4000 \
    --save_name "data/simulation/text_reuse"
```

3. To replicate candidate profile experiment (reuse):

```bash
python create_data/create_text.py \
    --prompt_file "data/candidate_profile/experiment" \
    --prompt_column "P" \
    --max_new_tokens 4096 \
    --task_type "repeat" \
    --save_hidden "data/candidate_profile/hidden_candidate" \
    --save_name "data/candidate_profile/text_candidate"
```


4. To replicate Hong Kong experiment Wave1 (reuse):

```bash
python create_data/create_text.py \
    --prompt_file "data/hongkong/HKData" \
    --prompt_column "text" \
    --max_new_tokens 4096 \
    --task_type "repeat" \
    --save_hidden "data/hongkong/hidden_HK" \
    --save_name "data/hongkong/text_HK"
```

5. To replicate Hong Kong experiment Wave2 (reuse):

```bash
python create_data/create_text.py \
    --prompt_file "data/hongkong/HKRepData" \
    --prompt_column "text" \
    --max_new_tokens 4096 \
    --task_type "repeat" \
    --save_hidden "data/hongkong/hidden_HKrep" \
    --save_name "data/hongkong/text_HKRep"
```

## Caution

If your purpose is not to replicate the paper exactly but just want to use our proposed method, we highly recommend using the latest model - LLaMa3 has a limited context window and the part of `create_data/create_text.py` is outdated due to the update in transformer library. See [our package website](https://gpi-pack.github.io/).

When you replicate the analysis, you need to create the directory to save the hidden states. Please refer to each shellscript (otherwise the coder will produce the error).

In addition, the slight difference in the computational environment (GPU and GPU driver) might lead to the difference in logit, which can cause the difference in the generated texts (this can be the problem of replicating `create_data/create_text.py` in the different environment). While the use of CPU can solve this problem, this is infeasible due to the computation time. Therefore, for the replication purpose, we also provide the generated texts (see `data/simulation/prompt_repeat.csv`).


## How to use LLaMa3 checkpoint (if you need to replicate exactly)

To use Llama3, you need to have an access to the model (otherwise you will encounter the error "Cannot access gated repo for url"). 
You can request the access from the following url: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Once you get access to the model, you also need to log in the huggingface.
You can log in by generating access token. To generate access token, go to "profile" -> "Settings" -> "Access Token".
See the following url for more details about access token: https://huggingface.co/docs/hub/en/security-tokens

You can pass your access token to this script in two ways:
1. Set your generated access token as a parameter of this script (access_token).
2. Add your access token to the environment variable (ACCESS_TOKEN). You can create ".env" file and add "ACCESS_TOKEN=<your_access_token>".
