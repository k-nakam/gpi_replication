# About this folder

This folder contains the files needed to replicate all the simulation studies in Section 4. This folder contains the following files:

1. `simulation/simulation.py`: Python code to run simulation. This simulation code has the following parameters:

2. `simulation/sim_util.py`: Python code that contains functions necessary to run simulations

Here are the arguments of `simulation/simulation.py`:

- `save_dir`: the directory to save the result (containing point estimate, standard error, and computation time for each method)

- `save_name`: the file name to be saved

- `task_type`: if you use the generated text from scratch, choose `create`. Otherwise (i.e., regenerate the input text), choose `repeat`

- `data_dir`: pkl file to be used (default is `data/simulation/text_processed`)

- `hidden_dir`: the directory where you saved hidden_states

- `nsim`: the number of simulations

- `noise_level`: the noise level of simulation (1.0 is used for the paper)

- `dgp_type`: the strength of confounding. `moderate` for $\alpha_1 = \alpha_2 = 10, \alpha_3 = \alpha_4 = 50$, `strong` for $\alpha_1 = \alpha_2 = 10, \alpha_3 = \alpha_4 = 100$, `strongest` for $\alpha_1 = \alpha_2 = 10, \alpha_3 = \alpha_4 = 1000$, `no_interaction` for $\alpha_1 = 10, \alpha_2 = 0, \alpha_3 = \alpha_4 = 50$, `no_interaction2` for $\alpha_1 = 10, \alpha_2 = 0, \alpha_3 = \alpha_4 = 100$, `no_interaction3` for $\alpha_1 = 10, \alpha_2 = 0, \alpha_3 = \alpha_4 = 1000$

- `confounding_type`: the variables used in the analysis. `confounding_2` for the analysis with separability and `confounding_4` for the analysis without separability.

- `test_size`: the size of data used for the statistical inference (0.5 is used for the paper)

- `method_use`: the string that determines which method to use. The first is OLS, the second is GPI, and the last is BERT-based method. For example, "[True, True, False]" means that the code estimates OLS and GPI, but not BERT.

- `save_ps`: whether you want to save the propensity score (needed to replicate Figure 4). YOU MUST CREATE THE DIRECTORY `simulation/ps/{dgp_type}_{confounding_type}` BEFORE YOU RUN.

# How to Use

For each data generating process and confounding type, you need to run the code. For example, if you want to run the analysis with strength $\alpha_1 = \alpha_2 = 10, \alpha_3 = \alpha_4 = 50$ and with separability for text reuse for just one time, here is the script you need to run:

```bash
python simulation/simulation.py \
    --data_dir "data/simulation/text_processed" \
    --hidden_dir "data/simulation/hidden_reuse" \
    --nsim 1 \
    --task_type "repeat"
```