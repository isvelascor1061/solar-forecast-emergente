import torch
import torch.nn as nn

class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM (many-to-one) with an optional automatic seq_len
    detection.

    Usage examples
    --------------
    # 1) Provide the sequence length explicitly
    model = BiLSTMRegressor(n_feat, hidden=64, seq_len=25,
                            num_layers=2, dropout=0.2)

    # 2) Let the model infer the sequence length from the first batch
    model = BiLSTMRegressor(n_feat, hidden=64,
                            num_layers=2, dropout=0.2)
    """
    def __init__(
        self,
        n_feat: int,
        hidden: int = 64,
        seq_len: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # --------------- Store hyper-parameters -----------------
        self.n_feat     = n_feat
        self.hidden     = hidden
        self.seq_len    = seq_len          # may be None → set inside forward
        self.num_layers = num_layers
        self.dropout_p  = dropout
        
        # -------------- LayerNorm (NEW) ------------------------
        self.pre_norm = nn.LayerNorm(n_feat)

        # --------------- Modules --------------------------------
        self.encoder = nn.LSTM(
            input_size    = n_feat,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0,
        )
        self.post_do = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden * 2, 1)   # ×2 because bidirectional

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L, n_feat)

        Returns
        -------
        Tensor (B,) – one prediction per sequence -> for the hour in the middle 
        """
        # Store sequence length once, if not preset
        if self.seq_len is None:
            self.seq_len = x.size(1)

        _, (h_n, _) = self.encoder(x)          # h_n: (num_layers*2, B, hidden)

        # Last layer, forward & backward directions
        h_fwd = h_n[-2]                        # (B, hidden)
        h_bwd = h_n[-1]                        # (B, hidden)

        pooled = self.post_do(torch.cat([h_fwd, h_bwd], dim=1))
        return torch.sigmoid(self.fc(pooled)).squeeze(-1)   # (B,) in (0, 1)
