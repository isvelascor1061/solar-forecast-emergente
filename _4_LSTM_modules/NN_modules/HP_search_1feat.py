#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to perform manual hyperparameter search for the LSTM model.
Replaces GridSearchCV (incompatible with 3D PyTorch sequences) with a
manual loop over the same param_grid combinations.
"""

import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from _4_LSTM_modules.NN_modules.first_LSTM_Model_sigmoid import LSTMRegressorsigmoid
import torch
import numpy as np
from torch.utils.data import DataLoader
from _4_LSTM_modules.Main_execution_files.LSTM_test_descaler_sigmoid import SeqDS, load_splits
from datetime import datetime


# -------------------------------------- Hyperparameter Search ----------------------------------

def grid_search():
    # Same param_grid as before
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'batch_size': [16, 32, 64, 128],
        'hidden_units': [32, 64, 128, 256],
        'dropout': [0.15, 0.3],
        'num_layers': [1, 2, 3, 4, 5]
    }

    # Load data splits
    SEQ_NPZ = "_4_LSTM_modules/Prepared_data/1feat_seq_sym13_CSI_shuffle.npz"
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total parameter combinations: {len(combinations)}")

    results = []
    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {params}")

        model = LSTMRegressorsigmoid(
            n_feat=X_tr.shape[2],
            hidden=params['hidden_units'],
            seq_len=X_tr.shape[1],
            multi_step=None,
            dropout=params['dropout'],
            num_layers=params['num_layers']
        )
        optim = torch.optim.AdamW(model.parameters(), lr=params['lr'])
        loss_fn = torch.nn.MSELoss()

        # Training loop (same 35 epochs as before)
        for epoch in range(35):
            model.train()
            for xb, yb in DataLoader(SeqDS(X_tr, y_tr), params['batch_size'], shuffle=True):
                optim.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optim.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32)
            y_pred = model(X_va_t).numpy().flatten()
        y_true = y_va.flatten()

        mse = mean_squared_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        results.append({**params, 'mse': mse, 'r2': r2})
        print(f"  MSE={mse:.4f}  R²={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values('mse')
    print("\nTop 5 combinations:")
    print(results_df.head())

    best_params = results_df.iloc[0][list(param_grid.keys())].to_dict()
    print(f"\nBest Parameters: {best_params}")
    return best_params


if __name__ == "__main__":
    best_params = grid_search()
    print(f"Optimal Hyperparameters: {best_params}")
