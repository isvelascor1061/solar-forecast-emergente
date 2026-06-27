#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AsymmetricLoss.py
=================
Asymmetric hour-weighted MSE loss for solar irradiance forecasting.

Motivation
----------
The BiLSTM model shows systematic zero-prediction failures at hours 8-10
(local time, Medellín): 424 test samples predict ~0 W/m² when GFS reports
clear-sky (kc ≈ 0.74) and observed mean GHI ≈ 460 W/m². The root cause is
min-loss hedging on the bimodal morning cloud distribution — standard MSE
rewards predicting the mean (≈0 when the distribution is bimodal with a
large zero-mass spike), rather than committing to the clear-sky branch.

This loss corrects that by:
  1. Penalising underestimation more than overestimation (alpha > 1).
  2. Giving extra weight to transition hours 8, 9, 10 where failures concentrate.
  3. Computing loss only on daytime samples (clear_sky_ghi > 0) — consistent
     with MSE_BASELINE_R, which is also a daytime-only metric.

Usage in LSTM_trainer.py
------------------------
    loss_fn = AsymmetricHourWeightedLoss(
        alpha=ALPHA,
        hour_weights=HOUR_WEIGHTS,
        use_daymask=True,
    )
    loss = loss_fn(preds, yb, hours_train[idx], mask_train[idx])
"""

import torch
import torch.nn as nn


class AsymmetricHourWeightedLoss(nn.Module):
    """
    Asymmetric MSE loss with hour-of-day weighting and daytime masking.

    Parameters
    ----------
    alpha : float
        Penalty multiplier for underestimation (y_true > y_pred).
        alpha=1.0 → symmetric MSE.  alpha=3.0 → underestimates cost 3×.
    hour_weights : dict[int, float]
        Extra multiplicative weight per local hour of day.
        Hours not listed default to 1.0.
        Example: {8: 1.5, 9: 2.5, 10: 2.5}
    use_daymask : bool
        If True, loss is computed only on samples where daymask=1
        (normalised by the count of daytime samples in the batch).
        Night samples (daymask=0) contribute 0 to both numerator and
        denominator, so they are completely ignored.

    Forward inputs
    --------------
    y_pred   : Tensor (B,) — model predictions in normalised space (CSI, 0-1)
    y_true   : Tensor (B,) — targets in normalised space (CSI, 0-1)
    hours    : LongTensor (B,) — local hour of day (0-23) for each sample
    daymask  : Tensor (B,) — 1.0 = daytime, 0.0 = night

    Returns
    -------
    Scalar loss tensor.
    """

    def __init__(
        self,
        alpha: float = 3.0,
        hour_weights: dict | None = None,
        use_daymask: bool = True,
    ) -> None:
        super().__init__()

        if hour_weights is None:
            hour_weights = {8: 1.5, 9: 2.5, 10: 2.5}

        self.alpha       = alpha
        self.use_daymask = use_daymask

        # Build a 24-element lookup table (one weight per hour).
        # Registered as a buffer so it moves to the correct device automatically.
        hour_w_table = torch.ones(24, dtype=torch.float32)
        for h, w in hour_weights.items():
            if not (0 <= h <= 23):
                raise ValueError(f"Hour key must be 0-23, got {h}")
            hour_w_table[h] = float(w)
        self.register_buffer("hour_w_table", hour_w_table)

    def forward(
        self,
        y_pred:  torch.Tensor,
        y_true:  torch.Tensor,
        hours:   torch.Tensor,
        daymask: torch.Tensor,
    ) -> torch.Tensor:
        # ── Squared error ─────────────────────────────────────────────────
        sq_err = (y_pred - y_true) ** 2

        # ── Asymmetric weight: underestimation costs alpha × more ─────────
        # under = 1 where y_true > y_pred (model missed the real value)
        under  = (y_true > y_pred).float()
        asym_w = 1.0 + (self.alpha - 1.0) * under   # 1.0 over, alpha under

        # ── Hour weight from lookup table ─────────────────────────────────
        hour_w = self.hour_w_table[hours]            # (B,)

        # ── Element-wise weighted loss ────────────────────────────────────
        loss_per = asym_w * hour_w * sq_err          # (B,)

        # ── Daytime mask ──────────────────────────────────────────────────
        if self.use_daymask:
            loss_per = loss_per * daymask
            n_day    = daymask.sum().clamp(min=1.0)  # avoid div-by-zero
            return loss_per.sum() / n_day
        else:
            return loss_per.mean()

    def extra_repr(self) -> str:
        hw = {i: float(self.hour_w_table[i]) for i in range(24)
              if float(self.hour_w_table[i]) != 1.0}
        return f"alpha={self.alpha}, hour_weights={hw}, use_daymask={self.use_daymask}"
