'''
Collection of functions

The contents are overlapped with TI_estimator/Qmod.py, but they are designed to return the pooled output.
These are used to calculate IOSS score in the empirical application.
'''

from __future__ import annotations

import math
import random
import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss

#from transformers import AdamW
from torch.optim import AdamW
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel
import os


import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

import sys
sys.path.append(f"{os.getcwd()}/src/TI_estimator")
sys.path.append(f"{os.getcwd()}/src")

from TIutil import *

import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
from TNutil import dml_score

import os

CUDA = (torch.cuda.device_count() > 0)
device = ("cuda" if torch.cuda.is_available() else "cpu")
MASK_IDX = 103


''' The first stage QNet'''
class CausalQNet(DistilBertPreTrainedModel):
    """ QNet model to estimate the conditional outcomes for the first stage
        Note the outcome Y is continuous """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.g_hat = nn.Linear(config.hidden_size, self.num_labels)
        self.Q_cls = nn.ModuleDict()
        for A in range(2):
          self.Q_cls['%d' % A] = nn.Sequential(
          nn.Linear(config.hidden_size, 200),
          nn.ReLU(),
          nn.Linear(200, 1))

        self.init_weights()

    def forward(self, text_ids, text_len, text_mask, A, Y, use_mlm=True):
        text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
        attention_mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
        mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
        target_words = torch.gather(text_ids, 1, mask)
        mlm_labels = torch.ones(text_ids.shape).long() * -100
        if CUDA:
            mlm_labels = mlm_labels.cuda()
        mlm_labels.scatter_(1, mask, target_words)
        text_ids.scatter_(1, mask, MASK_IDX)

        # distilbert output
        bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
        seq_output = bert_outputs[0]
        pooled_output = seq_output[:, 0]  # CLS token
        #print("bert_outputs:", bert_outputs)
        #print("seq_output:", seq_output.shape)
        #print("pooled_output:", pooled_output.shape)

        # masked language modeling objective
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = torch.nn.functional.gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss(reduction='sum')(prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = None

        # sm = nn.Softmax(dim=1)

        # A ~ text
        a_text = self.g_hat(pooled_output)
        a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.view(-1))
        # accuracy
        a_pred = a_text.argmax(dim=1)
        a_acc = (a_pred == A).sum().float()/len(A) 
        
        # Y ~ text+A
        # conditional expected outcomes
        Q0 = self.Q_cls['0'](pooled_output)
        Q1 = self.Q_cls['1'](pooled_output)

        if Y is not None:
            A0_indices = (A == 0).nonzero().squeeze()
            A1_indices = (A == 1).nonzero().squeeze()
            # Y loss
            y_loss_A1 = (((Q1.view(-1)-Y)[A1_indices])**2).sum()
            y_loss_A0 = (((Q0.view(-1)-Y)[A0_indices])**2).sum()
            y_loss = y_loss_A0 + y_loss_A1
        else:
            y_loss = 0.0

        return Q0, Q1, mlm_loss, y_loss, a_loss, a_acc, pooled_output

class QNet:
    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, a_weight = 1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None):

        self.model = CausalQNet.from_pretrained(
            'distilbert-base-uncased',
            num_labels = 2,
            output_attentions=False,
            output_hidden_states=False)

        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir
        self.losses = 0 #defined for hyperparameter tuning

    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,max_length=128,
                                                truncation=True,
                                                pad_to_max_length=True
                                                )

            out['text_id'].append(encoded_sent['input_ids'])
            out['text_mask'].append(encoded_sent['attention_mask'])
            out['text_len'].append(sum(encoded_sent['attention_mask']))
            out['A'].append(A)
            out['C'].append(C)
            out['Y'].append(Y)

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y'])
        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5):
        ''' Train the model'''

        # split data into two parts: one for training and the other for validation
        idx = list(range(len(texts)))
        random.shuffle(idx) # shuffle the index
        n_train = int(len(texts)*0.8) 
        n_val = len(texts)-n_train
        idx_train = idx[0:n_train]
        idx_val = idx[n_train:]

        # list of data
        train_dataloader = self.build_dataloader(texts[idx_train], 
            treatments[idx_train], confounders[idx_train], outcomes[idx_train])
        val_dataloader = self.build_dataloader(texts[idx_val], 
            treatments[idx_val], confounders[idx_val], outcomes[idx_val], sampler='sequential')
        

        self.model.train() 
        optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps=1e-8)

        best_loss = 1e20 #increased for the pusrpose of simulation (models are not properly stored)
        epochs_no_improve = 0

        for epoch in range(epochs):
            losses = []
            self.model.train()
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, _, Y = batch
            
                self.model.zero_grad()
                _, _, mlm_loss, y_loss, a_loss, a_acc, _ = self.model(text_id, text_len, text_mask, A, Y)

                # compute loss
                loss = self.loss_weights['a'] * a_loss + self.loss_weights['y'] * y_loss + self.loss_weights['mlm'] * mlm_loss
                
                       
                pbar.set_postfix({'Y loss': y_loss.item(),
                  'A loss': a_loss.item(), 'A accuracy': a_acc.item(), 
                  'mlm loss': mlm_loss.item()})

                # optimizaion for the baseline
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # evaluate validation set
            self.model.eval()
            pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
            a_val_losses, y_val_losses, a_val_accs = [], [], []
        
            for batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, _, Y = batch
                _, _, _, y_loss, a_loss, a_acc, _ = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
                
                a_val_losses.append(a_loss.item())
                y_val_losses.append(y_loss.item())

                # A accuracy
                a_acc = torch.round(a_acc*len(A))
                a_val_accs.append(a_acc.item())


            a_val_loss = sum(a_val_losses)/n_val
            print('A Validation loss:',a_val_loss)

            y_val_loss = sum(y_val_losses)/n_val
            print('Y Validation loss:',y_val_loss)

            val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
            print('Validation loss:',val_loss)

            a_val_acc = sum(a_val_accs)/n_val
            print('A accuracy:',a_val_acc)

            self.losses = val_loss
            # early stop
            if val_loss < best_loss:
                if self.modeldir and not os.path.exists(self.modeldir):
                    os.makedirs(self.modeldir)
                torch.save(self.model, f"{self.modeldir}/best_TI.pt") # save the best model
                best_loss = val_loss
                epochs_no_improve = 0              
            else:
                epochs_no_improve += 1
           
            if epoch >= 5 and epochs_no_improve >= patience:              
                print('Early stopping!' )
                print('The number of epochs is:', epoch)
                break

        # load the best model as the model after training
        self.model = torch.load(f"{self.modeldir}/best_TI.pt")

        return self.model

    def return_loss(self):
        #used for hyperparameter tuning
        return self.losses

    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get conditional expected outcomes Q0 and Q1 based on the training model'''
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = []  # E[Y|A=0, text]
        Q1s = []  # E[Y|A=1, text]
        pooled_outputs = []  # pooled output
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Statistics computing")

        if not model_dir:
            self.model.eval()
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _, pooled_output = self.model(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()
                pooled_outputs.append(pooled_output.detach().cpu().numpy())
        else:
            Qmodel = torch.load(model_dir)
            Qmodel.eval()
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = Qmodel(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()

        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]
        pooled_outputs = np.vstack(pooled_outputs)
        As = np.array(As)
        Ys = np.array(Ys)
        Cs = np.array(Cs)

        return Q0s, Q1s, As, Ys, Cs, pooled_outputs
        

''' The second stage: propensity scores estimation '''
def get_propensities(As, Q0s, Q1s, model_type='GaussianProcessRegression', kernel=None, random_state=0, n_neighbors=100, base_estimator=None):
    """Train the propensity model directly on the data 
    and compute propensities of the data"""

    Q_mat = np.array(list(zip(Q0s, Q1s)))

    if model_type == 'GaussianProcessRegression':
        if kernel == None:
            kernel = DotProduct() + WhiteKernel()
        propensities_mod = GaussianProcessClassifier(kernel=kernel, random_state=random_state, warm_start=True)
        propensities_mod.fit(Q_mat, As)

        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'KNearestNeighbors':
        propensities_mod = KNeighborsClassifier(n_neighbors=n_neighbors)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'DecisionTree':
        propensities_mod = DecisionTreeClassifier(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'AdaBoost':
        propensities_mod = AdaBoostClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Bagging':
        propensities_mod = BaggingClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Logistic':
        propensities_mod = LogisticRegression(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    return gs


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
    pooled_outputs = np.empty((len(Y), 768))
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
        Q0, Q1, A, Ys, _, pooled_output = mod.get_Q(X_test, T_test, np.zeros_like(T_test), Y_test)
        pooled_outputs[test_index, :] = pooled_output
     
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
    
    return ate_pryzant, se_pryzant, ate_gui, se_gui, time_pryzant, time_gui, pooled_outputs


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