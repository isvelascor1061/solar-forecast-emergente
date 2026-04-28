import torch
import torch.nn as nn


# =======================================================================
# BAHDANAU ATTENTION
# =======================================================================

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over a sequence of BiLSTM
    output vectors.

    For each time step t the mechanism computes an unnormalised scalar
    energy:
        e_t = v^T  tanh( W * h_t )

    where h_t is the BiLSTM output at step t and W, v are learnable
    parameters (implemented as a single Linear layer with bias).

    A softmax over all T steps produces the attention weights alpha_t
    (sum to 1).  The context vector is the weighted sum of all outputs:
        context = sum_t( alpha_t * h_t )

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector (= hidden * 2 for a BiLSTM).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        # Single linear layer that maps each time-step vector to a scalar
        # energy value; tanh is applied after to keep gradients bounded.
        self.score = nn.Linear(input_dim, 1, bias=True)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_outputs : Tensor (B, L, input_dim)
            Full output sequence produced by the BiLSTM.

        Returns
        -------
        context : Tensor (B, input_dim)
            Attention-weighted sum of encoder outputs.
        weights : Tensor (B, L)
            Normalised attention weights (each row sums to 1).
        """
        # Compute energy for every time step: (B, L, 1)
        energy = torch.tanh(self.score(encoder_outputs))

        # Softmax over the time dimension to get normalised weights
        weights = torch.softmax(energy, dim=1)          # (B, L, 1)

        # Weighted sum across the time dimension
        context = (weights * encoder_outputs).sum(dim=1)  # (B, input_dim)

        # Return weights without the trailing size-1 dimension
        return context, weights.squeeze(-1)              # (B, input_dim), (B, L)


# =======================================================================
# BI-LSTM + ATTENTION REGRESSOR
# =======================================================================

class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM with Bahdanau attention (many-to-one).

    Architecture
    ------------
    1. LayerNorm on the input features.
    2. BiLSTM encoder: produces one output vector per time step,
       shape (B, L, hidden * 2).
    3. BahdanauAttention: computes a context vector as the weighted
       sum of all time-step outputs.  Weights are returned so they
       can be visualised after training.
    4. Dropout + fully connected layer → scalar prediction.
    5. Optional sigmoid activation for targets normalised to [0, 1].

    Usage examples
    --------------
    # Standard forward pass (predictions only)
    out = model(x)                          # Tensor (B,)

    # Forward pass returning attention weights too
    out, weights = model(x, return_attention=True)
    # weights shape: (B, L)
    """

    def __init__(
        self,
        n_feat: int,
        hidden: int = 64,
        seq_len: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "sigmoid",
    ):
        super().__init__()

        # Store hyperparameters for reference
        self.n_feat     = n_feat
        self.hidden     = hidden
        self.seq_len    = seq_len          # may be None → inferred at first forward pass
        self.num_layers = num_layers
        self.dropout_p  = dropout
        self.activation = activation       # "sigmoid" or "linear"

        # Input normalisation: stabilises training and accelerates convergence
        self.pre_norm = nn.LayerNorm(n_feat)

        # BiLSTM encoder
        # Output shape: (B, L, hidden * 2)  — bidirectional doubles the width
        self.encoder = nn.LSTM(
            input_size    = n_feat,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0,
        )

        # Bahdanau attention over all L time-step outputs
        # Replaces the previous "take only last hidden state" approach
        self.attention = BahdanauAttention(input_dim=hidden * 2)

        # Regularisation before the output head
        self.post_do = nn.Dropout(dropout)

        # Output head: context vector (hidden * 2) → scalar
        self.fc = nn.Linear(hidden * 2, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, L, n_feat)
            Input sequence batch.
        return_attention : bool, default False
            When True, also return the attention weight matrix.

        Returns
        -------
        preds : Tensor (B,)
            One scalar prediction per sequence.
        weights : Tensor (B, L)   [only when return_attention=True]
            Attention weights for each time step.
        """
        # Infer sequence length on the first forward pass if not preset
        if self.seq_len is None:
            self.seq_len = x.size(1)

        # Layer normalisation on raw input features
        x = self.pre_norm(x)

        # BiLSTM: enc_out contains one vector per time step
        # We use enc_out (all steps) instead of only the final hidden state
        enc_out, _ = self.encoder(x)          # (B, L, hidden * 2)

        # Attention: compress the sequence into a single context vector
        context, attn_weights = self.attention(enc_out)  # (B, hidden*2), (B, L)

        # Dropout + linear projection
        out = self.fc(self.post_do(context)).squeeze(-1)  # (B,)

        # Output activation
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)           # bounded in (0, 1)

        if return_attention:
            return out, attn_weights
        return out
