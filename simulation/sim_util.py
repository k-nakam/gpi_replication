

from __future__ import annotations

import ast
import os
import numpy as np
import pandas as pd
import math
import time
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import matplotlib.pyplot as plt

import sys
#changed the following code accordingly
sys.path.append(f"{os.getcwd()}/src")
sys.path.append(f"{os.getcwd()}/src/TI_estimator")

from Qmod import QNet, get_propensities
from TNutil import dml_score

def dict_type(s):
    return ast.literal_eval(s)

def save_filename(filename: str, foldername: str) -> str:
    '''
    Function to find the filename without overwriting
    '''
    counter = 1
    new_filename = filename
    while os.path.isfile(f'{foldername}/{new_filename}.csv'):
        new_filename = f'{filename}({counter})'
        counter += 1
    filename = f'{foldername}/{new_filename}.csv'
    return filename


def set_DGP(
        T, R, X, data,
        noise_level: float = 1.0,
        dgp_dict = {"C": 100, "C2": 200, "T": 10},
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function to set data generating process for simulation

    Args:
    - T: np.ndarray, the treatment variable
    - R: np.ndarray, the extracted hidden representations
    - X: np.ndarray, the text data
    - data: pd.DataFrame, the dataset
    - noise_level: float, the noise level in data generating process, drawn from Gaussian Disritbution (default: 1.0)
    - dgp_dict: dict, the dictionary of the data generating process (default: {"C": 100, "C2": 200, "T": 10})
        - When you need to run interaction, you need to include them in pd.DataFrame

    Returns:
    - T: np.ndarray, the treatment variable
    - Y: np.ndarray, the outcome variable
    - R: np.ndarray, the extracted hidden representations
    - X: np.ndarray, the text data
    - dmat: np.ndarray, the design matrix
    '''

    dmat = []
    Y = np.random.normal(0, noise_level, size= len(T))
    for var_name, coef in dgp_dict.items():
        if var_name in data.columns:
            Y += coef * data[var_name].values
            dmat.append(data[var_name].values)
        else:
            raise ValueError(f"No such variable: '{var_name}' in the dataset")
    #create design matrix (for the subsequent OLS analyses)
    dmat = np.column_stack(dmat)

    return T, Y, R, X, dmat

def run_ols(Y : np.ndarray, T: np.ndarray, dmat: np.ndarray = None, simple: bool = True, verbose = True):
    '''
    Run OLS regression

    Args:
    - Y: np.ndarray, outcome variable
    - T: np.ndarray, treatment variable
    - dmat: np.ndarray, design matrix including both treatment and control variables (T, T:C1, C1, C2, C3, ...)
    - simple: bool, use of simple DGP (no interaction)
    - verbose: bool, whether to print the result
    '''
    if dmat is None:
        _, Y, _, T = train_test_split(Y, T, test_size = 0.5, random_state = 42)
    else:
        _, Y, _, T, _, dmat = train_test_split(Y, T, dmat, test_size = 0.5, random_state = 42)
    
    if dmat is None:
        control_status = "w/o control"
        dmat = sm.add_constant(T) #run regression without control
        lr = sm.OLS(Y, dmat).fit()
        #print(lr.summary())
        ate = lr.params[1]; se = lr.bse[1]
    else:
        dmat = sm.add_constant(dmat) #run regression with control
        lr = sm.OLS(Y, dmat).fit()
        #print(lr.summary())
        if simple:
            #no interaction models
            control_status = "w/ control (no interaction)"
            ate = lr.params[1]; se = lr.bse[1]
        else:
            #no interaction models
            control_status = "w/ control (interaction)"
            #with intraction -> need to compute marginal effect
            ate = (lr.params[1] + lr.params[2] * dmat[:,3]).mean() #\coef_{T} + \coef_{T:C1} * E[C1]
            # Calculate the standard error using the Delta method
            gradient = np.array([1, dmat[:,3].mean()]) #matrix of [1, E[C1]]
            cov_matrix = lr.cov_params()[1:3, 1:3] #V[\coef_{T}, \coef_{T:C1}]
            se = np.sqrt(gradient.T @ cov_matrix @ gradient)

    if verbose or verbose is None:
        print(f"ATE Estimate ({control_status}): {ate}\n")
        print(f"ATE (SE) ({control_status}): {se}\n")
    return ate, se


def TI_est(
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.5,
        batch_size: int = 8,
        lr: float = 2e-5,
        nepoch: int = 200,
        patience: int = 10,
        verbose:bool = True,
        plot_ps: bool = True,
        weight: dict = {"a_weight": 1.0, "y_weight": 1.0, "mlm_weight": 1.0},
        trim: list = [0.01, 0.99],
        save_ps: str = None,
        modeldir: str = None,
    ) -> tuple[float, float, float, float]:

    '''
    Implement Text-based estimator (Pryzant et al., 2021 / Gui and Veitch 2023)

    Args
    - X: np.ndarray, text data
    - T: np.ndarray, treatment variable
    - Y: np.ndarray, outcome variable
    - test_size: float, the percentage of validation data (default: 0.2)
    - batch_size: int, the batch size (default: 8)
    - lr: float, the learning rate (default: 2e-5)
    - nepoch: int, the number of epochs (default: 200)
    - patience: int, the patience for early stopping (default: 10)
    - verbose: bool, whether to print the result (default: True)
    - plot_ps: bool, whether to plot the propensity score (default: True)
    - save_ps: str, the directory of saving propensity score plots (default: None)
    - modeldir: str, the directory of saving the model (default: None)
    '''

    start_time = time.time()
    # K-fold cross fitting
    train_index, test_index = train_test_split(np.arange(len(Y)), test_size=test_size)

    X_train, X_test = X[train_index], X[test_index]
    T_train, T_test = T[train_index], T[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    mod = QNet(
            batch_size = batch_size,
            a_weight = weight['a_weight'], # weight for treatment prediction
            y_weight = weight['y_weight'], # weight for outcome prediction
            mlm_weight=weight['mlm_weight'], # weight for masked language modeling
            modeldir=modeldir,
    )
    mod.train(X_train, T_train, np.zeros_like(T_train), Y_train, epochs=nepoch, learning_rate = lr, patience = patience)
    Q0, Q1, A, Ys, _ = mod.get_Q(X_test, T_test, np.zeros_like(T_test), Y_test)
    
    #pryzant
    psi = np.array(Q1) - np.array(Q0)
    ate_pryzant = psi.mean()
    se_pryzant = psi.std() / math.sqrt(len(psi))

    time_pryzant = time.time() - start_time

    #gui and veitch
    g = get_propensities(A, Q0, Q1,
            model_type='GaussianProcessRegression', # choose the nonparametric model
            kernel=None,    # kernel function for GPR
            random_state=0
        ) # random seed for GPR

    if plot_ps: #Plot Propensity Score
        _ = plt.hist(g[A == 1], alpha = 0.5, label = "Treated", density = True)
        _ = plt.hist(g[A == 0], alpha = 0.5, label = "Control", density = True)
        plt.xlabel("Estimated Propensity Score")
        plt.ylabel("Density")
        plt.legend(loc = "upper left")
        plt.show()

    if save_ps is not None: #Save propensity score and treatment variables
        ps_data = pd.DataFrame({'A': T_test, 'g': g})
        file_name = save_filename('PS_TI', save_ps)
        ps_data.to_csv(file_name, index=False)

    if trim is not None:
        g[g > max(trim)] = max(trim)
        g[g < min(trim)] = min(trim)

    psi2 = dml_score(A, Ys, g, np.array(Q1), np.array(Q0))  # error bound for confidence interval)

    ate_gui = psi2.mean()
    se_gui = psi2.std() / math.sqrt(len(psi2))
    time_gui = time.time() - start_time

    if verbose or verbose is None:
        print(f"Modeling Texts (Pryzant et al. 2021): {ate_pryzant} / Confidence Interval: [{ate_pryzant - 1.96 * se_pryzant}, {ate_pryzant + 1.96 * se_pryzant}]\n")
        print(f"Modeling Texts (Gui and Veitch 2023): {ate_gui} / Confidence Interval: [{ate_gui - 1.96 * se_gui}, {ate_gui + 1.96 * se_gui}]\n")
        print(f"Elapsed Time (Pryzant et al. 2021): {time_pryzant} seconds\n")
        print(f"Elapsed Time (Gui and Veitch 2023): {time_gui} seconds\n")
    return ate_pryzant, se_pryzant, ate_gui, se_gui, time_pryzant, time_gui


def TI_est_K(
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        K: int = 2,
        batch_size: int = 8,
        lr: float = 2e-5,
        nepoch: int = 200,
        patience: int = 10,
        verbose:bool = True,
        save_ps: str = None,
        plot_ps: bool = True,
        trim: list = [0.01, 0.99],
        modeldir: str = None,
    ) -> tuple[float, float, float, float]:

    '''
    Implement Text-based estimator (Pryzant et al., 2021 / Gui and Veitch 2023)
    with K-fold cross fitting

    Args
    - X: np.ndarray, text data
    - T: np.ndarray, treatment variable
    - Y: np.ndarray, outcome variable
    - test_size: float, the percentage of validation data (default: 0.2)
    - batch_size: int, the batch size (default: 8)
    - lr: float, the learning rate (default: 2e-5)
    - nepoch: int, the number of epochs (default: 200)
    - patience: int, the patience for early stopping (default: 10)
    - verbose: bool, whether to print the result (default: True)
    - plot_ps: bool, whether to plot the propensity score (default: True)
    - save_ps: str, the directory of saving propensity score plots (default: None)
    - modeldir: str, the directory of saving the model (default: None)
    '''

    start_time = time.time()
    
    # K-fold cross fitting
    psi_list = []; psi2_list = []; time_pryzant = []; time_gui = []
    kf = KFold(n_splits=K, shuffle=True)
    
    for train_index, test_index in kf.split(Y):
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        mod = QNet(
            batch_size = batch_size,
            a_weight = 1.0, # weight for treatment prediction
            y_weight = 1.0, # weight for outcome prediction
            mlm_weight=1.0, # weight for masked language modeling
            modeldir=modeldir,
        )
        # Note: Do not add confounding information (as it is optional)
        mod.train(X_train, T_train, np.zeros_like(T_train), Y_train, epochs=nepoch, learning_rate = lr, patience = patience)
        Q0, Q1, A, Ys, _ = mod.get_Q(X_test, T_test, np.zeros_like(T_test), Y_test)
     
        #pryzant
        psi = np.array(Q1) - np.array(Q0)
        psi_list.extend(psi)
        time_pryzant.append(time.time() - start_time)
        
        #gui and veitch
        g = get_propensities(A, Q0, Q1,
            model_type='GaussianProcessRegression', # choose the nonparametric model
            kernel=None,    # kernel function for GPR
            random_state=0
        ) # random seed for GPR
        
        if plot_ps: #Plot Propensity Score
            _ = plt.hist(g[A == 1], alpha = 0.5, label = "Treated", density = True)
            _ = plt.hist(g[A == 0], alpha = 0.5, label = "Control", density = True)
            plt.xlabel("Estimated Propensity Score")
            plt.ylabel("Density")
            plt.legend(loc = "upper left")
            plt.show()
        
        if trim is not None:
            g[g > max(trim)] = max(trim)
            g[g < min(trim)] = min(trim)
        
        if save_ps is not None: #Save propensity score and treatment variables
            ps_data = pd.DataFrame({'A': T_test, 'g': g})
            file_name = save_filename('PS_TI', save_ps)
            ps_data.to_csv(file_name, index=False)
        
        psi2 = dml_score(A, Ys, g, np.array(Q1), np.array(Q0))  # error bound for confidence interval)
        psi2_list.extend(psi2)
        time_gui.append(time.time() - start_time)
            
    ate_pryzant = np.mean(psi_list)
    se_pryzant = np.std(psi_list) / math.sqrt(len(psi_list))
    time_pryzant = np.sum(time_pryzant)
    
    ate_gui = np.mean(psi2_list)
    se_gui = np.std(psi2_list) / math.sqrt(len(psi2_list))
    time_gui = np.sum(time_gui)

    if verbose or verbose is None:
        print(f"Modeling Texts (Pryzant et al. 2021): {ate_pryzant} / Confidence Interval: [{ate_pryzant - 1.96 * se_pryzant}, {ate_pryzant + 1.96 * se_pryzant}]\n")
        print(f"Modeling Texts (Gui and Veitch 2023): {ate_gui} / Confidence Interval: [{ate_gui - 1.96 * se_gui}, {ate_gui + 1.96 * se_gui}]\n")
        print(f"Elapsed Time (Pryzant et al. 2021): {time_pryzant} seconds\n")
        print(f"Elapsed Time (Gui and Veitch 2023): {time_gui} seconds\n")
    
    return ate_pryzant, se_pryzant, ate_gui, se_gui, time_pryzant, time_gui