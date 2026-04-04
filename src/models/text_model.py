import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Transformer-based webpage text encoder.
    Supports both BERT and DistilBERT via AutoModel.
    Uses the [CLS] token representation as the text embedding.
    """

    def __init__(self, config: dict):
        super().__init__()
        text_cfg = config["text"]
        self.model_name = text_cfg["model_name"]
        self.output_dim = text_cfg["output_dim"]
        self.dropout_p = text_cfg["dropout"]
        self.freeze_layers = text_cfg.get("freeze_layers", 4)

        # Load pre-trained model (works for both BERT and DistilBERT)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        # Freeze early layers to save memory
        self._freeze_layers(self.freeze_layers)

        self.dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, self.output_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.output_dim * 2, self.output_dim),
        )
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def _freeze_layers(self, n_layers: int):
        """Freeze embeddings and first n_layers transformer blocks."""
        # Freeze embeddings
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        # DistilBERT uses encoder.transformer.layer, BERT uses encoder.encoder.layer
        if hasattr(self.encoder, "transformer"):
            # DistilBERT
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, "encoder"):
            # BERT
            layers = self.encoder.encoder.layer
        else:
            return

        for i, layer in enumerate(layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        out = self.dropout(cls_output)
        out = self.projection(out)
        out = self.layer_norm(out)
        return out
