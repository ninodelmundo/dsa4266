import torch
import torch.nn as nn
from typing import Optional

from ..data.data_utils import URL_VOCAB_SIZE


class URLEncoder(nn.Module):
    """
    Character-level URL encoder.
    Supports three architectures: CNN, LSTM, or small Transformer.
    """

    def __init__(self, config: dict):
        super().__init__()
        url_cfg = config["url"]
        self.vocab_size = URL_VOCAB_SIZE  # use actual computed vocab size
        self.embedding_dim = url_cfg["embedding_dim"]
        self.hidden_dim = url_cfg["hidden_dim"]
        self.output_dim = url_cfg["output_dim"]
        self.model_type = url_cfg["model_type"]
        self.max_length = url_cfg["max_length"]
        self.dropout_p = url_cfg["dropout"]
        self.num_layers = url_cfg.get("num_layers", 2)

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=0
        )

        if self.model_type == "cnn":
            self._build_cnn()
        elif self.model_type == "lstm":
            self._build_lstm()
        elif self.model_type == "transformer":
            self._build_transformer()
        else:
            raise ValueError(f"Unknown URL model type: {self.model_type}")

        self.dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Linear(self._encoder_output_dim(), self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    # CNN Architecture

    def _build_cnn(self):
        kernel_sizes = [3, 5, 7]
        num_filters = self.hidden_dim // len(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.embedding_dim, num_filters, k, padding=k // 2
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )
        self._cnn_output_dim = num_filters * len(kernel_sizes)

    def _build_lstm(self):
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0,
        )
        self._lstm_output_dim = self.hidden_dim * 2  # bidirectional

    def _build_transformer(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        self._transformer_output_dim = self.embedding_dim

    def _encoder_output_dim(self) -> int:
        if self.model_type == "cnn":
            return self._cnn_output_dim
        elif self.model_type == "lstm":
            return self._lstm_output_dim
        elif self.model_type == "transformer":
            return self._transformer_output_dim

    # Forward

    def forward(self, url_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            url_tokens: (batch, max_length) long tensor
        Returns:
            embedding: (batch, output_dim)
        """
        x = self.embedding(url_tokens)  # (B, L, E)

        if self.model_type == "cnn":
            xt = x.permute(0, 2, 1)  # (B, E, L) for Conv1d
            pooled = [conv(xt).max(dim=-1).values for conv in self.convs]
            out = torch.cat(pooled, dim=-1)  # (B, cnn_output_dim)

        elif self.model_type == "lstm":
            out, _ = self.lstm(x)       # (B, L, 2H)
            out = out.mean(dim=1)       # mean over all timesteps (better than last-only)

        elif self.model_type == "transformer":
            # Create padding mask
            padding_mask = url_tokens == 0  # (B, L) True where padded
            out = self.transformer_encoder(
                x, src_key_padding_mask=padding_mask
            )
            out = out.mean(dim=1)  # mean pooling

        out = self.dropout(out)
        out = self.projection(out)
        out = self.layer_norm(out)
        return out


def build_url_encoder(config: dict) -> URLEncoder:
    """Factory function to build a URLEncoder from config."""
    return URLEncoder(config)
