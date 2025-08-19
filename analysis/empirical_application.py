
'''
Replication Code for real world application in
"Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments"

Author: Kentaro Nakamura (Email: knakamura@g.harvard.edu)
'''
from __future__ import annotations

import pandas as pd
import numpy as np
import os
import argparse
import sys

sys.path.append(f"{os.getcwd()}/src")
sys.path.append(f"{os.getcwd()}/simulation")
from application_util import TI_est_K
from TarNet import estimate_k
from TNutil import load_hiddens
import time
import torch
from scipy.spatial.distance import cdist

torch.manual_seed(1337)
np.random.seed(1337)

# Helper function to clean text data
def clean_text(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    return str(t)

parser = argparse.ArgumentParser(description='Empirical Application for CausalText')
parser.add_argument('--K', type=int, default=2, help="The number of cross-fitting folds")
parser.add_argument('--application', type=str, required=True, choices=["candidate_profile", "hongkong1", "hongkong2"], help="Type of application to run")
parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the data")
parser.add_argument('--hidden_dir', type=str, required=True, help="Directory containing hidden states")
args = parser.parse_args()

if __name__ == '__main__':
    #############################
    # STEP 1: Loading Variables (based on fine-tuning results)
    #############################
    # Load the data
    df = pd.read_csv(f"{args.data_dir}.csv").sort_index()
    print(f"Sample size is {len(df)}")

    if args.application == "candidate_profile":
        varlist = ["T", "X", "resp"] #T, X, Y
        config_deconfounder = {
            'lr': 9.0e-6,
            'dropout': 0.15,
        }
        config_TI = {
            'lr': 5.8e-6,
        }
        
    elif args.application == "hongkong1":
        varlist = ["treatycommit", "text", "resp"] #T, X, Y
        config_deconfounder = {
            'lr': 2e-5,
            'dropout': 0.10,
        }
        config_TI = {
            'lr': 2e-5,
        }
    elif args.application == "hongkong2":
        varlist = ["treatycommit", "text", "resp"] #T, X, Y
        config_deconfounder = {
            'lr': 2e-5,
            'dropout': 0.10,
        }
        config_TI = {
            'lr': 2e-5,
        }
    else:
        raise ValueError("Invalid application type. Choose from 'candidate_profile', 'hongkong1', or 'hongkong2'.")

    # loading each variable. Z will be no longer used as we remove regularization
    T = df[varlist[0]].values

    # X (text data)
    X = df[varlist[1]].values
    X = [clean_text(t) for t in X]
    X = np.array(X)
    
    # R (hidden states)
    hidden_index = range(len(df))
    R = load_hiddens(args.hidden_dir, hidden_list = hidden_index, prefix = "hidden_last_")
    
    # Y (outcome variable)
    Y = df[varlist[2]].values

    #############################
    # STEP 2: Run each analysis
    #############################  
    # The proposed method
    start_time = time.time()
    ate1, se1, err_list, fr_list = estimate_k(
        R = R, 
        Y = Y, 
        T = T, 
        K = args.K, 
        model_dir = f"analysis/best_mod/{args.application}/",
        valid_perc = 0.2, 
        plot_propensity = True, 
        trim = [0.01, 0.99], 
        step_size = 500,
        batch_size = 32, 
        nepoch = 500, 
        lr = config_deconfounder['lr'],
        architecture_y = [500, 1], 
        architecture_z = [2048],
        dropout = config_deconfounder['dropout'], 
        bn = False, 
        save_ps = f"analysis/ps/{args.application}/",
        patience = 15,
    )
    time1 = time.time() - start_time
    print("GPI:", ate1, "/ Confidence Interval: [", ate1 - 1.96 * se1, ",", ate1 + 1.96 * se1, "]\n" )
    print("Elapsed Time:", time1, "seconds\n")
    
    # IOSS score (Hausdorff distance between f(R)|T=1 and f(R)|T=0)
    pair_scores = []
    n, fr_d = fr_list.shape
    
    for j in range(fr_d):
        fr_j = fr_list[:, j:j+1]
        fr_j_std = (fr_j - fr_j.min()) / (fr_j.max() - fr_j.min() + 1e-12)
        
        fr_j_std1 = fr_j_std[T == 1]
        fr_j_std0 = fr_j_std[T == 0]

        d01 = cdist(fr_j_std1, fr_j_std0, metric="euclidean").min(axis=1)
        d10 = cdist(fr_j_std0, fr_j_std1, metric="euclidean").min(axis=1)
        max01 = d01.max()
        max10 = d10.max()
        pair_scores.append(max(max01, max10))

    ioss_score = np.mean(pair_scores)

    print("IOSS score:", ioss_score, "\n")
    
    # TI-estimator
    pryzant_coef, pryzant_se, gui_coef, gui_se, time2, time3, pooled_outputs = TI_est_K(
        X = X, 
        T = T, 
        Y = Y, 
        K = args.K, 
        patience= 15,
        batch_size= 32, 
        nepoch = 500, 
        lr = config_TI['lr'],
        trim = [0.01, 0.99],
        plot_ps = True, 
        save_ps = f"analysis/ps/{args.application}/",
        modeldir= f"analysis/best_mod/{args.application}/",
    )
    
    # IOSS score (Hausdorff distance between f(R)|T=1 and f(R)|T=0)
    pair_scores = []
    n, pooled_outputs_d = pooled_outputs.shape

    for j in range(pooled_outputs_d):
        pooled_j = pooled_outputs[:, j:j+1]
        pooled_j_std = (pooled_j - pooled_j.min()) / (pooled_j.max() - pooled_j.min() + 1e-12)

        pooled_j_std1 = pooled_j_std[T == 1]
        pooled_j_std0 = pooled_j_std[T == 0]

        d01 = cdist(pooled_j_std1, pooled_j_std0, metric="euclidean").min(axis=1)
        d10 = cdist(pooled_j_std0, pooled_j_std1, metric="euclidean").min(axis=1)
        max01 = d01.max()
        max10 = d10.max()
        pair_scores.append(max(max01, max10))

    ioss_score2 = np.mean(pair_scores)

    print("IOSS score:", ioss_score2, "\n")