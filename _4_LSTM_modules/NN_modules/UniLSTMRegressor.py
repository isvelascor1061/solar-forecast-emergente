import torch
import torch.nn as nn


# =======================================================================
# BAHDANAU ATTENTION  (identical to BiLSTMRegressor version)
# =======================================================================

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over an LSTM output sequence.

    For each time step t:
        e_t = v^T  tanh( W * h_t )

    Softmax over all T steps → attention weights alpha_t (sum to 1).
    Context vector = weighted sum of all time-step outputs.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector.
        For a unidirectional LSTM this equals `hidden`  (not hidden * 2).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1, bias=True)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_outputs : Tensor (B, L, input_dim)

        Returns
        -------
        context : Tensor (B, input_dim)
        weights : Tensor (B, L)
        """
        energy  = torch.tanh(self.score(encoder_outputs))  # (B, L, 1)
        weights = torch.softmax(energy, dim=1)              # (B, L, 1)
        context = (weights * encoder_outputs).sum(dim=1)   # (B, input_dim)
        return context, weights.squeeze(-1)                 # (B, input_dim), (B, L)


# =======================================================================
# UNIDIRECTIONAL LSTM + ATTENTION REGRESSOR
# =======================================================================

class UniLSTMRegressor(nn.Module):
    """
    Unidirectional LSTM with Bahdanau attention (many-to-one).

    Identical to BiLSTMRegressor except:
      - bidirectional=False  →  LSTM output is (B, L, hidden), not (B, L, hidden*2)
      - BahdanauAttention receives hidden-dim vectors (not hidden*2)
      - FC layer: hidden → 1  (not hidden*2 → 1)

    This makes the model ~3× smaller than the BiLSTM equivalent at the
    same hidden size, while retaining the same attention mechanism.

    Architecture
    ------------
    1. LayerNorm on input features.
    2. Unidirectional LSTM encoder → (B, L, hidden).
    3. BahdanauAttention → context vector (B, hidden).
    4. Dropout + FC → scalar prediction.
    5. Optional sigmoid for targets normalised to [0, 1].

    Usage
    -----
    out         = model(x)                          # Tensor (B,)
    out, weights = model(x, return_attention=True)  # also returns (B, L)
    """

    def __init__(
        self,
        n_feat: int,
        hidden: int = 128,
        seq_len: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.25,
        activation: str = "sigmoid",
    ):
        super().__init__()

        self.n_feat     = n_feat
        self.hidden     = hidden
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.dropout_p  = dropout
        self.activation = activation

        # Input normalisation
        self.pre_norm = nn.LayerNorm(n_feat)

        # Unidirectional LSTM — output shape: (B, L, hidden)
        self.encoder = nn.LSTM(
            input_size    = n_feat,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = False,          # key difference from BiLSTMRegressor
            dropout       = dropout if num_layers > 1 else 0,
        )

        # Attention over (B, L, hidden) — input_dim = hidden (not hidden * 2)
        self.attention = BahdanauAttention(input_dim=hidden)

        self.post_do = nn.Dropout(dropout)

        # Output head: hidden → 1  (not hidden * 2 → 1)
        self.fc = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, L, n_feat)
        return_attention : bool

        Returns
        -------
        preds   : Tensor (B,)
        weights : Tensor (B, L)   [only when return_attention=True]
        """
        if self.seq_len is None:
            self.seq_len = x.size(1)

        x = self.pre_norm(x)                           # (B, L, n_feat)
        enc_out, _ = self.encoder(x)                   # (B, L, hidden)
        context, attn_weights = self.attention(enc_out) # (B, hidden), (B, L)
        out = self.fc(self.post_do(context)).squeeze(-1) # (B,)

        if self.activation == "sigmoid":
            out = torch.sigmoid(out)

        if return_attention:
            return out, attn_weights
        return out
