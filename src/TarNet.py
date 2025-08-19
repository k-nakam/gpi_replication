
'''
Code for Estimating Treatment Effect using TarNet
Aurhor: Kentaro Nakamura (knakamura@g.harvard.edu)

Last Update: June 28th, 2024
'''

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Union

from TNutil import (
    TarNetBase,
    TarNet_loss,
    estimate_psi_split,
)

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

class TarNet:
    '''
    Wrapper class for TarNet model

    Attributes:
    - self.device: torch.device, device used for training
    - self.epochs: int, number of epochs
    - self.batch_size: int, batch size
    - self.num_workers: int, number of workers for data loader
    - self.train_dataloader: DataLoader, dataloader for training
    - self.valid_dataloader: DataLoader, dataloader for validation
    - self.model: TarNetBase, TarNet model
    - self.optim: torch.optim, optimizer (Adam by default)
    - self.scheduler: torch.optim.lr_scheduler, learning rate scheduler

    Methods:
    - create_dataloaders: create dataloaders for training and validation
    - fit: fit the TarNet model
    - validate_step: validate the model
    - predict: predict the outcome without pertubation
    - pert_predict: predict the outcome with pertubation
    '''
    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        architecture_y: list = [1],
        architecture_z: list = [1024],
        dropout: float = 0.3,
        step_size: int = None,
        bn: bool = False,
        patience: int = 5,
        min_delta: float = 0.01,
        model_dir: str = None,
        model_id: str = "best_TarNet",
        verbose: bool = True,
    ):
        '''
        Initializers of the class

        Args:
        - epochs: int, number of epochs
        - batch_size: int, batch size
        - learning_rate: float, learning rate
        - architecture_y: list, architecture of the outcome model
        - architecture_z: list, architecture of the shared representation model
        - dropout: float, dropout rate
        - step_size: int, step size for the learning rate scheduler (if None, no scheduler)
        - bn: bool, whether to use batch normalization
        - patience: int, patience for early stopping
        - min_delta: float, minimum delta for early stopping
        - model_dir: str, directory for saving the model
        - model_id: str, unique id to save the model (default: "best_TarNet")
        - verbose: bool, whether to print the device
        '''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: Using {self.device}")
        self.epochs = epochs; self.batch_size = batch_size
        self.train_dataloader = None; self.valid_dataloader = None
        self.model = TarNetBase(
            sizes_z = architecture_z,
            sizes_y = architecture_y,
            dropout=dropout,
            bn=bn,
        ).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.step_size = step_size
        if self.step_size is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)
        self.loss_f = TarNet_loss
        self.valid_loss = 0
        self.patience = patience
        self.min_delta = min_delta
        self.model_dir = model_dir
        self.model_id = model_id
        if self.model_dir is not None:
             if not os.path.exists(self.model_dir):
                 print(f"The directory {self.model_dir} does not exist.")
        self.verbose = verbose

    def create_dataloaders(self,
            r_train: Union[np.ndarray, torch.Tensor],
            r_test: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            y_test: Union[np.ndarray, torch.Tensor],
            t_train: Union[np.ndarray, torch.Tensor],
            t_test: Union[np.ndarray, torch.Tensor],
        ):
        '''
        Create dataloader for training and validation

        Args:
        - r_train: np.array or torch.Tensor, training data for internal representation
        - r_test: np.array or torch.Tensor, test data for internal representation
        - y_train: np.array or torch.Tensor, training data for outcome
        - y_test: np.array or torch.Tensor, test data for outcome
        - t_train: np.array or torch.Tensor, training data for treatment
        - t_test: np.array or torch.Tensor, test data for treatment
        '''
        inputs = [r_train, r_test, y_train, y_test, t_train, t_test]
        for i, input in enumerate(inputs):
            if isinstance(input, np.ndarray):
                inputs[i] = torch.Tensor(input)
            elif not isinstance(input, torch.Tensor):
                raise ValueError("Input must be either numpy array or torch.Tensor")

            if i > 1: #for y, t
                inputs[i] = inputs[i].reshape(-1, 1)
        r_train, r_test, y_train, y_test, t_train, t_test = inputs

        train_dataset = TensorDataset(r_train, t_train, y_train)
        valid_dataset = TensorDataset(r_test, t_test, y_test)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler = RandomSampler(train_dataset))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler = SequentialSampler(valid_dataset))

    def fit(self,
            R: Union[np.ndarray, torch.Tensor],
            Y: Union[np.ndarray, torch.Tensor],
            T: Union[np.ndarray, torch.Tensor],
            valid_perc: float = None, 
            plot_loss: bool = True,
        ):
        '''
        Fit the TarNet model

        Args:
        - R: np.array or torch.Tensor, internal representation
        - Y: np.array or torch.Tensor, outcome
        - T: np.array or torch.Tensor, treatment
        - valid_perc: float, percentage of validation data (from 0 to 1)
        - plot_loss: bool, whether to plot the training and validation loss
        '''

        R_train, R_test, Y_train, Y_test, T_train, T_test = train_test_split(
            R, Y, T, test_size=valid_perc, random_state= 42,
        )
        self.create_dataloaders(R_train, R_test, Y_train, Y_test, T_train, T_test) #r, y, t
        all_training_loss = []; all_valid_loss = []; best_loss = 1e10; epochs_no_improve = 0

        #training loop
        self.model.train()
        for epoch in range(self.epochs):
            loss_list = []
            if self.verbose:
                pbar = tqdm(total=len(self.train_dataloader), desc=f'Training (Epoch {epoch})')

            for _, (r, t, y) in enumerate(self.train_dataloader): #r, t, y
                if self.verbose:
                    pbar.update()
                self.optim.zero_grad()
                y0_pred, y1_pred, _, _ = self.model(r.to(self.device)) #y0, y1, eps, z
                loss = self.loss_f(
                    y_true = y.to(self.device), 
                    t_true = t.to(self.device), 
                    y0_pred = y0_pred, 
                    y1_pred = y1_pred,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping
                self.optim.step()
                loss_list.append(loss.item() / len(y))
            self.model.eval()
            valid_loss = self.validate_step()
            all_training_loss.append(np.mean(loss_list)); all_valid_loss.append(valid_loss)
            if self.verbose:
                print(f"epoch: {epoch}--------- train_loss: {np.mean(loss_list)} ----- valid_loss: {valid_loss}")
            self.model.train()
            if self.step_size is not None:
                self.scheduler.step(valid_loss) #when using ReduceLROnPlateau
            
            #save the best model
            if valid_loss + self.min_delta < best_loss:
                if self.model_dir != "" and self.model_dir is not None:
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)
                    torch.save(self.model.state_dict(), f"{self.model_dir}/{self.model_id}.pth")
                best_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            #early stopping options
            if epoch >= 5 and epochs_no_improve >= self.patience:
                print(f"Early stopping! The number of epoch is {epoch}.")
                break
        
        if self.model_dir != "" and self.model_dir is not None:
            if self.verbose:
                print(f"Loading the model saved at {self.model_dir}...")
            self.model.load_state_dict(torch.load(f"{self.model_dir}/{self.model_id}.pth", weights_only=True))
        
        if plot_loss:
            _ = plt.plot(all_training_loss, label = "Training Loss")
            _ = plt.plot(all_valid_loss, label = "Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def validate_step(self) -> torch.Tensor:
        '''
        Method to Validate the model
        '''
        valid_loss = []
        with torch.no_grad():
            if self.verbose:
                pbar = tqdm(total=len(self.valid_dataloader), desc= f'Validating')
            for _, (r, t, y) in enumerate(self.valid_dataloader): #r t y
                if self.verbose:
                    pbar.update()
                y0_pred, y1_pred, _, _ = self.model(r.to(self.device)) #y0, y1, eps, fr
                loss = self.loss_f(
                    y_true = y.to(self.device), 
                    t_true = t.to(self.device), 
                    y0_pred = y0_pred, 
                    y1_pred = y1_pred,
                )
                valid_loss.append(loss / len(y))
            print(f"Difference: {torch.mean(y1_pred - y0_pred)}")
        self.valid_loss = torch.Tensor(valid_loss).mean()
        return self.valid_loss

    def predict(self, r: Union[np.ndarray, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Predict method for TarNet

        Args:
        r: np.ndarray or torch.Tensor, internal representation

        Returns:
        y0_preds: torch.Tensor, predicted outcome for control
        y1_preds: torch.Tensor, predicted outcome for treated
        frs: torch.Tensor, deconfounder
        '''
        if isinstance(r, np.ndarray):
            r = torch.Tensor(r)
        elif not isinstance(r, torch.Tensor):
            raise ValueError("Input must be either numpy array or torch.Tensor")
        
        #create dataloader for batching
        dataset = TensorDataset(r)
        dataloader  = DataLoader(dataset, batch_size= self.batch_size)

        y0_preds = []; y1_preds = []; frs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, 
                              total=len(dataloader),
                              desc = 'Predicting'): #r
                y0_pred, y1_pred, _, fr = self.model(batch[0].to(self.device)) #y0, y1, eps, fr
                y0_preds.append(y0_pred)
                y1_preds.append(y1_pred)
                frs.append(fr)
        
        # Concatenate all the batched predictions
        y0_preds = torch.cat(y0_preds)
        y1_preds = torch.cat(y1_preds)
        frs = torch.cat(frs)
        
        return y0_preds, y1_preds, frs

def estimate_k(
        R: Union[np.ndarray, list],
        Y: Union[np.ndarray, list],
        T: Union[np.ndarray, list],
        K: int = 2,
        valid_perc: float = 0.2,
        plot_propensity: bool = True,
        ps_model = RandomForestClassifier,
        ps_model_params: dict = {},
        batch_size: int = 32,
        nepoch: int = 200,
        step_size: int = None,
        lr: float = 2e-5,
        dropout: float = 0.2,
        architecture_y: list = [1],
        architecture_z: list = [2048],
        trim: list = [0.01, 0.99],
        bn: bool = False,
        patience: int = 5,
        min_delta: float = 0,
        save_ps: str = None,
        model_dir: str = None,
        verbose: bool = True,
    ) -> tuple[float, float, float]:
    '''
    The function estimates the ATE using TarNet and returns the estimated ATE, standard deviation, and the validation loss (with k-fold cross-fitting)
    
    Args:
    - R: list or np.ndarray, list of covariates (obtained from the hidden states)
    - Y: list or np.ndarray, list of outcomes
    - T: list or np.ndarray, list of treatments
    - K: int, number of folds (default: 2)
    - valid_perc: float, percentage of validation data (default: 0.2)
    - plot_propensity: bool, whether to plot the propensity score (default: True)
    - ps_model: sklearn model, propensity score model (default: RandomForestClassifier)
    - ps_model_params: dict, parameters for the propensity score model (default: {})
    - batch_size: int, batch size (default: 64)
    - nepoch: int, number of epochs (default: 200)
    - step_size: int, step size for the learning rate scheduler (default: None)
    - lr: float, learning rate (default: 1e-3)
    - dropout: float, dropout rate (default: 0.3)
    - trim: list, trimming range for the propensity score (e.g., [0.05,0.95]) (default: [0.01, 0.99])
    - architecture_y: list, architecture of the outcome model (default: [1])
    - architecture_z: list, architecture of the shared representation model (default: [2048])
    - bn: bool, whether to use batch normalization (default: True)
    - patience: int, patience for early stopping (default: 5)
    - min_delta: float, minimum delta for early stopping (default: 0.01)
    - save_ps: str, the directory of saving propensity score plots (default: None)
    - model_dir: str, directory for saving the model (default: None)
    - verbose: bool, whether to print the device (default: True)
    '''
    
    inputs = [R,Y,T]
    
    for i, input in enumerate(inputs):
        if isinstance(input, list):
            inputs[i] = np.array(input)
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be either numpy array or list, but not ", type(input))
    
    R, Y, T = inputs

    psi_list = []
    fr_list = np.zeros((len(Y), architecture_z[-1]))
    err_list = np.zeros((len(Y)))
    kf = KFold(n_splits=K, shuffle=True)
    
    for train_index, test_index in kf.split(Y):
        r_train, r_test = R[train_index], R[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        t_train, t_test = T[train_index], T[test_index]
        model = TarNet(
            epochs= nepoch, 
            learning_rate = lr, 
            batch_size= batch_size,
            architecture_y = architecture_y, 
            architecture_z = architecture_z, 
            dropout=dropout,
            step_size= step_size, bn=bn,
            patience= patience, 
            min_delta= min_delta, 
            model_dir=model_dir, 
            verbose=verbose,
        )
        model.fit(R = r_train, Y = y_train, T = t_train, valid_perc = valid_perc)
        y0_pred, y1_pred, fr = model.predict(r_test)
        fr_list[test_index, :] = fr.cpu().numpy()
        
        # store the error
        y_pred = y1_pred.cpu().numpy().reshape(-1) * t_test.reshape(-1) + y0_pred.cpu().numpy().reshape(-1) * (1 - t_test.reshape(-1))
        err = y_test.reshape(-1) - y_pred.reshape(-1)
        err_list[test_index] = err

        print(f"T-learner: {np.mean(y1_pred.cpu().numpy() - y0_pred.cpu().numpy())}")
    
        psi, tpreds = estimate_psi_split(fr = fr, t = t_test, y = y_test, y0 = y0_pred, y1 = y1_pred, 
                           plot_propensity = plot_propensity, trim = trim,
                           ps_model= ps_model, ps_model_params= ps_model_params)
        psi_list.extend(psi)
    
        #the following function is for the simulation purpose (saving the propensity score for plotting)
        if save_ps is not None: #Save propensity score and treatment variables
            ps_data = pd.DataFrame({'T': t_test, 'tpreds': tpreds})
            file_name = save_filename('PS_TarNet', save_ps)
            ps_data.to_csv(file_name, index=False)    
    
    ate_est = np.mean(psi_list)
    sd_est = np.std(psi_list) / np.sqrt(len(psi_list))
    
    print("ATE:", ate_est, " /  SE:", sd_est)

    return ate_est, sd_est, err_list, fr_list


def estimate_and_loss(
        R: Union[np.ndarray, list],
        Y: Union[np.ndarray, list],
        T: Union[np.ndarray, list],
        test_size: float = 0.5,
        valid_perc: float = 0.2,
        plot_propensity: bool = True,
        ps_model = RandomForestClassifier,
        ps_model_params: dict = {},
        batch_size: int = 32,
        nepoch: int = 200,
        step_size: int = None,
        lr: float = 2e-5,
        dropout: float = 0.2,
        architecture_y: list = [1],
        architecture_z: list = [2048],
        trim: list = [0.01, 0.99],
        bn: bool = False,
        patience: int = 5,
        min_delta: float = 0.01,
        save_ps: str = None,
        task_type: str = "create",
        model_dir: str = None,
        unique_id: str = None,
        verbose: bool = True,
    ) -> tuple[float, float, float, list[float]]:
    '''
    Note: This function is only for simulation to test the performance of the estimator.
    
    The function estimates the ATE using TarNet and returns the estimated ATE and the standard deviation (no cross-fitting)
    '''
    
    inputs = [R,Y,T]
    
    for i, input in enumerate(inputs):
        if isinstance(input, list):
            inputs[i] = np.array(input)
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be either numpy array or list, but not ", type(input))
    R, Y, T = inputs

    train_index, test_index = train_test_split(np.arange(len(Y)), test_size=test_size)

    r_train, r_test = R[train_index], R[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    t_train, t_test = T[train_index], T[test_index]
    model = TarNet(
            epochs= nepoch, 
            learning_rate = lr, 
            batch_size= batch_size,
            architecture_y = architecture_y, 
            architecture_z = architecture_z, 
            dropout=dropout,
            step_size= step_size, 
            bn=bn,
            patience= patience, 
            min_delta= min_delta, 
            model_dir=model_dir, 
            model_id=unique_id, 
            verbose=verbose,
        )
    model.fit(R = r_train, Y = y_train, T = t_train, valid_perc = valid_perc)
    y0_pred, y1_pred, fr = model.predict(r_test)
    
    #estimate propensity score
    print("Cross-fitting propensity score...")
    psi, tpreds = estimate_psi_split(
        fr = fr, t = t_test, y = y_test, y0 = y0_pred, y1 = y1_pred, 
        plot_propensity = plot_propensity, trim = trim,
        ps_model= ps_model, ps_model_params= ps_model_params
    )
    
    #the following function is for the simulation purpose (saving the propensity score for plotting)
    if save_ps is not None: #Save propensity score and treatment variables
        ps_data = pd.DataFrame({'T': t_test, 'tpreds': tpreds})
        file_name = save_filename(filename = 'PS_TarNet' + task_type, foldername=save_ps)
        ps_data.to_csv(file_name, index=False)
    
    ate_est = np.mean(psi)
    se_est = np.std(psi) / np.sqrt(len(psi))

    return ate_est, se_est, tpreds