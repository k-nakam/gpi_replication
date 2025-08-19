"""
Collection of functions
"""

from __future__ import annotations

import torch
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

class TarNetBase(nn.Module):
    def __init__(self,
            sizes_z: tuple = [200, 200, 200],
            sizes_y: tuple = [200, 100, 100, 1],
            dropout: float=None,
            bn: bool = False
        ):
        '''
        Class for TarNet model (PyTorch). First layer is Transformer architecture

        Args:
            sizes_z: tuple, size of hidden layers for shared representation
            sizes_y: tuple, size of hidden layers for outcome prediction
            dropout: float, dropout rate (default: 0.3)
            bn: bool, whether to use batch normalization (default: False)
                Note that after the first layer everything is the feedforward network.
        '''

        super(TarNetBase, self).__init__()
        self.bn: bool = bn
        self.model_z = self._build_model(sizes_z, dropout) #model for shared representation
        self.model_y1 = self._build_model(sizes_y, dropout) #model for Y(1)
        self.model_y0 = self._build_model(sizes_y, dropout) #model for Y(0)

        #model for epsilon (legacy, to be removed)
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def _build_model(self, sizes: tuple, dropout: float) -> nn.Sequential:
        # create model by nn.Sequential
        layers = []
        for out_size in sizes:
            layers.append(nn.LazyLinear(out_features=out_size))
            if self.bn:
                layers.append(nn.BatchNorm1d(out_size, track_running_stats=False))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        if self.bn and dropout is not None:
            layers = layers[:-3]  # remove the last BN, ReLU and Dropout
        elif self.bn == False and dropout is None:
            layers = layers[:-1] # remove the last ReLU
        else:
            layers = layers[:-2] # remove the last ReLU and Dropout
        return nn.Sequential(*layers)
    
    def forward(
            self, 
            inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        z = self.model_z(inputs)
        z_trans = nn.functional.relu(z)
        y0 = self.model_y0(z_trans)
        y1 = self.model_y1(z_trans)
        eps = self.epsilon(torch.ones_like(y1)[:, 0:1])
        return y0, y1, eps, z

def dml_score(t, y, tpred, ypred1, ypred0) -> np.ndarray:
    '''
    Calculate influence function for the average treatment effect

    Args:
    - t: np.ndarray or torch.Tensor, treatment
    - y: np.ndarray or torch.Tensor, outcome
    - tpred: np.ndarray or torch.Tensor, predicted treatment
    - ypred1: np.ndarray or torch.Tensor, predicted outcome when treated
    - ypred0: np.ndarray or torch.Tensor, predicted outcome when untreated

    Returns:
    - psi: np.array, influence function for the average treatment effect
    '''

    inputs = [t, y, tpred, ypred1, ypred0]
    for i, input in enumerate(inputs):
        if isinstance(input, torch.Tensor):
            inputs[i] = input.cpu().numpy()
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be a numpy array or a PyTorch tensor")
    t, y, tpred, ypred1, ypred0 = inputs

    t = t.reshape(-1); y = y.reshape(-1); tpred = tpred.reshape(-1); ypred1 = ypred1.reshape(-1); ypred0 = ypred0.reshape(-1)

    psi = ypred1 - ypred0 + t * (y - ypred1) / tpred - (1 - t) * (y - ypred0) / (1 - tpred)

    return psi


def estimate_psi_split(
        fr,
        t,
        y,
        y0,
        y1,
        ps_model = RandomForestClassifier,
        ps_model_params: dict = {},
        trim: list = [0.01, 0.99],
        plot_propensity: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

    '''
    Estimate Propensity Score from latent representation with cross-fitting

    Args:
    - fr: np.ndarray or torch.Tensor, latent representation
    - t: np.ndarray or torch.Tensor, treatment
    - y: np.ndarray or torch.Tensor, outcome
    - y0: np.ndarray or torch.Tensor, outcome when treated
    - y1: np.ndarray or torch.Tensor, outcome when untreated
    - ps_model: propensity score model (default: GaussianProcessClassifier)
    - ps_model_params: dict, hyperparameters for the propensity score model
    - trim: list, trimming quantiles for propensity score (default: [0.01, 0.99])
    - plot_propensity: bool, whether to plot the propensity score (default: False)

    Returns:
    psi: np.array, influence function for the average treatment effect
    '''
    inputs = [fr, t, y, y0, y1]
    for i, input in enumerate(inputs):
        if isinstance(input, torch.Tensor):
            inputs[i] = input.cpu().numpy()
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be a numpy array or a PyTorch tensor")
    fr, t, y, y0, y1 = inputs
    
    model_train = ps_model(**ps_model_params)
    model_test = ps_model(**ps_model_params)
    
    ind = np.arange(fr.shape[0])
    ind1, ind2 = train_test_split(ind, test_size=0.5, random_state=42)

    model_train.fit(fr[ind1,:], t[ind1])
    tpred2 = model_train.predict_proba(fr[ind2,:])[:,1]
    acc2 = accuracy_score(t[ind2], tpred2.round())
    
    model_test.fit(fr[ind2,:], t[ind2])
    tpred1 = model_test.predict_proba(fr[ind1,:])[:,1]
    acc1 = accuracy_score(t[ind1], tpred1.round())
    
    acc = (acc1 + acc2) / 2
    print(f"Accuracy Score of Propensity Score Model: {acc}")
    
    tpreds = np.zeros(len(t))
    tpreds[ind1] = tpred1; tpreds[ind2] = tpred2
    
    if plot_propensity:
        ts = np.zeros(len(t))
        ts[ind1] = t[ind1]; ts[ind2] = t[ind2]
        _ = plt.hist(tpreds[ts == 1], alpha = 0.5, label = "Treated", density = True)
        _ = plt.hist(tpreds[ts == 0], alpha = 0.5, label = "Control", density = True)
        plt.xlabel("Estimated Propensity Score")
        plt.ylabel("Density")
        plt.legend(loc = "upper left")
        plt.show()

    if trim is not None:
        tpreds[tpreds < min(trim)] = min(trim)
        tpreds[tpreds > max(trim)] = max(trim)    
    
    psi1 = dml_score(t[ind1], y[ind1], tpreds[ind1], y1[ind1], y0[ind1])
    psi2 = dml_score(t[ind2], y[ind2], tpreds[ind2], y1[ind2], y0[ind2])
    psi = np.append(psi1, psi2)
    return psi, tpreds


def load_hiddens(directory: str, hidden_list: list, prefix: str = 'hidden_last_', device: torch.device = "cpu") -> torch.Tensor:
    """
    Function to load hidden representations given a list of file names (without the .pt extension).
    The order of the loaded tensors will follow the order of hidden_list.

    Args:
    - directory: str, directory where hidden representations are stored
    - hidden_list: list of str, list of file names (without .pt) to load
    - prefix: str, prefix of the file names (default: None)
    - device: torch.device, device to load the tensors (default: cpu)

    Returns:
    - tensors: torch.Tensor, 3D tensor of hidden representations (batch_size, seq_len, hidden_dim)
    """
    # Determine map_location
    tensors = []
    for name in tqdm(hidden_list, desc="Loading Tensors"):
        if prefix is not None:
            file_path = os.path.join(directory, f"{prefix}{name}.pt")
        else:
            file_path = os.path.join(directory, f"{name}.pt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        tensor = torch.load(file_path, map_location=device, weights_only=True).float().cpu()
        tensors.append(tensor)

    tensors = torch.stack(tensors, dim=0).squeeze(1).numpy()
    return tensors


def TarNet_loss(
        y_true : torch.Tensor,
        t_true : torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
    ) -> torch.Tensor:
    '''
    Calculate loss function for TarNet with Pertubation
    
    Args:
    - y_true: torch.Tensor, true outcome
    - t_true: torch.Tensor, true treatment
    - y0_pred: torch.Tensor, predicted outcome when untreated
    - y1_pred: torch.Tensor, predicted outcome when treated
    '''
    
    with torch.no_grad():
        T0_indices = (t_true.view(-1) == 0).nonzero().squeeze()
        T1_indices = (t_true.view(-1) == 1).nonzero().squeeze()
    
    #vanilla loss (original TarNet loss)
    loss0 = ( ((y0_pred.view(-1)-y_true.view(-1))[T0_indices])**2).sum()
    loss1 = ( ((y1_pred.view(-1)-y_true.view(-1))[T1_indices])**2).sum()
    loss_y = loss0 + loss1

    return loss_y

