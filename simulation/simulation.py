'''
Replication Code for simulation in
"Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments"

Author: Kentaro Nakamura (Email: knakamura@g.harvard.edu)
'''
from __future__ import annotations

import pandas as pd
import numpy as np
import os
import argparse
import ast
import sys
import datetime

# Change here accordingly
sys.path.append(f"{os.getcwd()}/src")
sys.path.append(f"{os.getcwd()}/simulation")

from sim_util import set_DGP, run_ols, TI_est
from TarNet import estimate_and_loss
from TNutil import load_hiddens
import time
import torch

torch.manual_seed(1337)
np.random.seed(1337)

def simulation(
        T, R, X, data,
        config_deconfounder: dict,
        config_TI: dict,
        dgp_dict = {"T": 10, "TC1": 50, "C1": 50, "C2": 50, "C3": 50},
        noise_level: float = 1.0,
        test_size:float = 0.5,
        trim: list = [0.01, 0.99],
        method_use: list[bool, bool, bool] = [False, True, False],
        dgp_type: str = None,
        save_ps: bool = False,
        unique_id: str = "best_TarNet",
        task_type: str = "create",
        confounding_type: str = None,
    ) -> tuple[float]:
    '''
    Code to run simulation

    Args:
    - T: np.ndarray, the treatment variable
    - R: np.ndarray, the extracted hidden representations
    - X: np.ndarray, the text data
    - data: pd.DataFrame, the dataset
    - dgp_dict: dict, the dictionary of the data generating process ({ "Variable Name": str, "Coefficient": float})
    - test_size: float, the percentage of validation data (default: 0.5)
    - batch_size: list[int,int,int,int,int], list of batch sizes for each estimator
    - nepoch: int, the number of epochs (default: 200)
    - lr: list[float,float,float,float,float], list of learning rates for each estimator
    - dgp_type: str, the type of data generating process, used to store the propensity score plots (default: None)
    - save_ps: bool, whether to save the propensity score plots (default: False)
    - unique_id: str, the name to save the model (default: None)
    - task_type: str, either create or repeat (used to differentiate the saved propensity score)
    - confounding_type: str, the level of complexity of confounding (weak, moderate, or strong) (default: None)
    '''

    #loadting data
    T, Y, R, X, dmat = set_DGP(T = T, R = R, X = X, data = data, dgp_dict = dgp_dict, noise_level=noise_level)
    
    if args.save_ps:
        save_ps = "simulation/ps/" + dgp_type + "_" + confounding_type
    else:
        save_ps = None

    #OLS (for baseline)
    if method_use[0]:
        start_time = time.time()
        ols_coef1, ols_se1 = run_ols(Y = Y, T = T, dmat = None)
        time0 = time.time() - start_time
        if dmat is not None:
            ols_coef2, ols_se2 = run_ols(Y = Y, T = T, dmat = dmat, simple = False) #make simple = True if no interaction
        else:
            ols_coef2, ols_se2 = 0, 0
    else:
        ols_coef1, ols_se1, ols_coef2, ols_se2 = 0, 0, 0, 0
    
    #Deconfounder (GPI method)
    if method_use[1]:
        start_time = time.time()
        ate1, se1, _ = estimate_and_loss(
            R = R, 
            Y = Y, 
            T = T, 
            test_size = test_size,
            model_dir = f"simulation/best_mod/{dgp_type}_{confounding_type}/",
            valid_perc = 0.2, 
            plot_propensity = True,
            trim = trim, 
            step_size = config_deconfounder['step_size'],
            batch_size = config_deconfounder['batch_size'], 
            nepoch = config_deconfounder['nepoch'], 
            lr = config_deconfounder['lr'],
            architecture_y = config_deconfounder['architecture_y'], 
            architecture_z = config_deconfounder['architecture_z'],
            dropout = config_deconfounder['dropout'], 
            bn = config_deconfounder['bn'], 
            save_ps = save_ps,
            task_type=task_type,
            patience = config_deconfounder['patience'],
            unique_id = unique_id,
        )
        time1 = time.time() - start_time
        print("Deconfounder:", ate1, "/ Confidence Interval: [", ate1 - 1.96 * se1, ",", ate1 + 1.96 * se1, "]\n" )
        print("Elapsed Time:", time1, "seconds\n")
    else:
        ate1, se1, time1 = 0, 0, 0

    #TI-estimator
    if method_use[2]:
        pryzant_coef, pryzant_se, gui_coef, gui_se, time2, time3 = TI_est(
            X, 
            T, 
            Y, 
            test_size = test_size, 
            patience= config_TI['patience'],
            batch_size= config_TI['batch_size'], 
            nepoch = config_TI['nepoch'], 
            lr = config_TI['lr'], 
            weight = config_TI['weight'],
            trim = trim, 
            plot_ps = True, 
            save_ps = save_ps, 
            modeldir= f"simulation/best_mod/{dgp_type}_{confounding_type}/",
        )
    else:
        pryzant_coef, pryzant_se, gui_coef, gui_se, time2, time3 = 0, 0, 0, 0, 0, 0

    return ols_coef1, ols_se1, ols_coef2, ols_se2, ate1, se1, pryzant_coef, pryzant_se, gui_coef, gui_se, time0, time1, time2, time3


parser = argparse.ArgumentParser(description='Experiments for CausalText')
parser.add_argument('--save_dir', type=str, default="simulation/result/")
parser.add_argument('--save_name', type=str, default="result")
parser.add_argument('--task_type', type=str, default="create", help="The type of task (create or reuse)")
parser.add_argument('--data_dir', type=str, default="data/simulation/text_processed")
parser.add_argument('--hidden_dir', type=str, default="data/hidden_create")
parser.add_argument('--nsim', type=int, default = 100)
parser.add_argument('--noise_level', type=float, default = 1.0)
parser.add_argument('--dgp_type', type=str, default = 'moderate', help = "Type of data generating process (strong or moderate confounding)")
parser.add_argument('--confound_type', type=str, default = 'confounding2', help = "Level of complexity of confounding")
parser.add_argument('--test_size', type=float, default = 0.5, help = "The size of test data used for the estimation")
parser.add_argument('--method_use', type=str, default = "[True, True, True]", help = "For development purposes (do not use it in the actual simulation)")
parser.add_argument('--save_ps', type=bool, default = True, help = "Save propensity score values in csv format or not")
args = parser.parse_args()

if __name__ == '__main__':
    # select the type of weight for Pryzant / Gui and Veitch
    weight = {"a_weight": 1.0, "y_weight": 1.0, "mlm_weight": 1.0}
    
    # load the configuration of hyperparameters (based on fine-tuning)
    if args.dgp_type  == "moderate":
        lr_value = 1.6e-5 if args.task_type == "create" else 2.0e-5
        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 5e-5,
            "patience": 15,
            "weight": weight,
        }
    elif args.dgp_type == "strong":
        lr_value = 1.6e-5 if args.task_type == "create" else 5.0e-5
        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 2e-5,
            "patience": 15,
            "weight": weight,
        }
    elif args.dgp_type == "strongest":
        lr_value = 2.4e-5 if args.task_type == "create" else 4.5e-5
        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 9e-6,
            "patience": 15,
            "weight": weight,
        }
    elif args.dgp_type == "no_interaction":
        #lr value
        if args.task_type == "create":
            lr_value = 1e-5
        else:
            lr_value = 4e-5

        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 2e-5,
            "patience": 15,
            "weight": weight,
        }
    elif args.dgp_type == "no_interaction2":
        #lr value
        if args.task_type == "create":
            lr_value = 2e-5
        else:
            lr_value = 2.8e-5

        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 2e-5,
            "patience": 15,
            "weight": weight,
        }
    elif args.dgp_type == "no_interaction3":
        #lr value
        if args.task_type == "create":
            lr_value = 3e-5
        else:
            lr_value = 3.5e-5
        config_deconfounder = {
            "step_size": 500,
            "batch_size": 32,
            "nepoch": 500,
            "lr": lr_value,
            "architecture_y": [500, 1],
            "architecture_z": [2048],
            "dropout": 0.15,
            "bn": False,
            "patience": 15,
        }
        config_TI = {
            "batch_size": 32,
            "nepoch": 500,
            "lr": 5e-5,
            "patience": 15,
            "weight": weight,
        }
    
    # Set DGP
    if args.confound_type == "confounding2": #topic-model based confounding with no overlap
        var_list = ["T", "TC1_2", "C1_2", "C2"]
        sign = -1
    elif args.confound_type == "confounding4": #topic-model based confounding with overlap
        var_list = ["T_4", "T_4C1_4", "C1_4", "C2"]
        sign = -1
    else:
        raise("No such confounding type. Please select either 'confounding2' or 'confounding4'")
    
    if args.dgp_type  == "moderate":
        dgp_dict = {var_list[0]: 10, var_list[1]: 10, var_list[2]: -50, var_list[3]: sign*50}   
    elif args.dgp_type == "strong":
        dgp_dict = {var_list[0]: 10, var_list[1]: 10, var_list[2]: -100, var_list[3]: sign*100}
    elif args.dgp_type == "strongest":
        dgp_dict = {var_list[0]: 10, var_list[1]: 10, var_list[2]: -1000, var_list[3]: sign*1000}
    elif args.dgp_type == "no_interaction":
        dgp_dict = {var_list[0]: 10, var_list[1]: 0, var_list[2]: -100, var_list[3]: sign*100}
    elif args.dgp_type == "no_interaction2":
        dgp_dict = {var_list[0]: 10, var_list[1]: 0, var_list[2]: -50, var_list[3]: sign*50}
    elif args.dgp_type == "no_interaction3":
        dgp_dict = {var_list[0]: 10, var_list[1]: 0, var_list[2]: -1000, var_list[3]: sign*1000}
    else:
        raise("No such DGP type. Please select either 'moderate', 'strong', 'strongest', 'no_interaction', 'no_interaction2', 'no_interaction3'")
    method_use = ast.literal_eval(args.method_use)

    # Load data
    if not os.path.isfile(f"{args.data_dir}.pkl"):
        raise FileNotFoundError(f"No such file: '{args.data_dir}.pkl'")

    df = pd.read_pickle(f"{args.data_dir}.pkl").sort_index()
    print(f"Sample size is {len(df)}")

    T = df[var_list[0]].values
    X = df["X"].values
    hidden_index = df["index"].values
    R = load_hiddens(args.hidden_dir, hidden_list = hidden_index)
    print(f"Hidden representation shape: {R.shape}")

    store = np.zeros([args.nsim, 14]) #store result

    unique_id = "best_TarNet" +  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(args.nsim):
        out = simulation(
            T = T, 
            R = R, 
            X = X, 
            data = df,
            config_deconfounder=config_deconfounder,
            config_TI=config_TI, 
            dgp_dict= dgp_dict, 
            noise_level=args.noise_level,
            test_size = args.test_size,
            method_use= method_use, 
            dgp_type=args.dgp_type, 
            save_ps=args.save_ps, 
            unique_id = unique_id,
            confounding_type = args.confound_type,
            task_type= args.task_type,
        )
        store[i] = out #store simulation results
        
    #store results
    os.chdir(args.save_dir)
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    
    pd.DataFrame(store).to_csv(f"{args.save_name}_{args.dgp_type}_{args.confound_type}_{date_time}.csv")