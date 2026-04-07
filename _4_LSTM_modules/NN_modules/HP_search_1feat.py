#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to perform GridSearchCV for hyperparameter tuning on LSTM model.
It will search for optimal hyperparameters for the LSTMRegressor model.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from _4_LSTM_modules.NN_modules.first_LSTM_Model_sigmoid import LSTMRegressorsigmoid
from _4_LSTM_modules.NN_modules.Losses import MixedHuberMSELoss, RampMSELoss, WeightedMSELoss, MixedLoss
import torch
import numpy as np
from torch.utils.data import DataLoader
from _4_LSTM_modules.Main_execution_files.LSTM_test_descaler_sigmoid import SeqDS, load_splits
from datetime import datetime

# ---------------------------------- Wrapper for LSTM Model ------------------------------------
# Custom Wrapper for LSTM to make it compatible with GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class LSTMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, lr=1e-3, batch_size=32, hidden_units=64, dropout=0.15, num_layers=1):
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.num_layers = num_layers

    def fit(self, X, y):
        # Create LSTM model with given hyperparameters
        model = LSTMRegressorsigmoid(n_feat=X.shape[2], 
                                      hidden=self.hidden_units, 
                                      seq_len=X.shape[1], 
                                      multi_step=None, 
                                      dropout=self.dropout, 
                                      num_layers=self.num_layers)
        optim = torch.optim.AdamW(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        for epoch in range(35):  # Change epochs based on your requirement
            model.train()
            for xb, yb in DataLoader(SeqDS(X, y), self.batch_size, shuffle=True):
                optim.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optim.step()

        self.model = model  # Save the trained model
        return self

    def predict(self, X):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)  # Umwandlung in Tensor
            for xb in X_tensor:
                pred = self.model(xb.unsqueeze(0))  # Um sicherzustellen, dass die Dimension stimmt
                predictions.append(pred.detach().numpy().flatten())  # Füge flache (1D) Vorhersagen hinzu
    
        return np.array(predictions)  # Jetzt eine zusammengeführte Liste von Vorhersagen




# -------------------------------------- Hyperparameter Search ----------------------------------

def grid_search():
    # Set the param_grid for GridSearchCV
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],  # Learning rates to test
        'batch_size': [16, 32, 64, 128],  # Batch sizes to test
        'hidden_units': [32, 64, 128, 256],  # Number of hidden units
        'dropout': [0.15, 0.3],  # Dropout rate
        'num_layers': [1, 2, 3, 4, 5]  # Number of layers in LSTM
    }

    # Load data splits
    SEQ_NPZ = "_4_LSTM_modules/Prepared_data/1feat_seq_sym13_CSI_shuffle.npz"
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)

    # Create a GridSearchCV object using the LSTMRegressorWrapper
    grid_search = GridSearchCV(LSTMRegressorWrapper(), param_grid, scoring='r2', cv=2, verbose=2)

    print(f"Total parameter combinations: {len(param_grid['lr']) * len(param_grid['batch_size']) * len(param_grid['hidden_units']) * len(param_grid['dropout']) * len(param_grid['num_layers'])}")
    # Perform GridSearch with custom progress reporting
    grid_search.fit(X_tr, y_tr)

    # Output the best hyperparameters
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_params_


if __name__ == "__main__":
    best_params = grid_search()
    print(f"Optimal Hyperparameters: {best_params}")
